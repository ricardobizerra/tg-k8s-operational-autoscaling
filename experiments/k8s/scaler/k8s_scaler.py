import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Tuple

import kubernetes
import kubernetes.client
import kubernetes.config

from config import ScalerConfig

logger = logging.getLogger(__name__)


@dataclass
class ScalingEvent:
  timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
  namespace: str = ""
  model_name: str = ""
  dataset_slug: str = ""
  scaling_mode: str = ""
  current_replicas: int = 1
  desired_replicas: int = 1
  current_memory_mb: int = 0
  desired_memory_mb: int = 0
  predicted_cpu_cores: float = 0.0
  actual_cpu_cores: float = 0.0
  predicted_ram_gb: float = 0.0
  actual_ram_gb: float = 0.0
  action_horizontal: str = "no_change"  # scale_up | scale_down | no_change | cooldown
  action_vertical: str = "no_change"
  sla_cpu_violation: bool = False
  sla_ram_violation: bool = False
  selected_model: str = ""
  fit_time_cpu_seconds: float = 0.0
  predict_time_cpu_seconds: float = 0.0
  fit_time_ram_seconds: float = 0.0
  predict_time_ram_seconds: float = 0.0
  refit_skipped: int = 0
  cycle_number: int = 0
  phase: str = "evaluation"
  pod_restart_count: int = 0
  pod_ready_latency_seconds: float = 0.0


def _parse_memory_quantity(q: str) -> int:
  if not q:
    return 0
  q = q.strip()
  if q.endswith("Mi"):
    return int(q[:-2])
  if q.endswith("Gi"):
    return int(float(q[:-2]) * 1024)
  if q.endswith("Ki"):
    return max(1, int(q[:-2]) // 1024)
  if q.endswith("M"):
    return int(q[:-1])
  if q.endswith("G"):
    return int(float(q[:-1]) * 1000)
  return max(1, int(q) // (1024 * 1024))


class KubernetesScaler:
  def __init__(self, config: ScalerConfig) -> None:
    try:
      kubernetes.config.load_incluster_config()
      logger.info("Using in-cluster Kubernetes config")
    except kubernetes.config.ConfigException:
      kubernetes.config.load_kube_config()
      logger.info("Using local kubeconfig")

    self._apps = kubernetes.client.AppsV1Api()
    self._cfg = config

    self._last_scale_up_h: float = 0.0
    self._last_scale_down_h: float = 0.0
    self._last_scale_v: float = 0.0
    self._start_time: float = time.monotonic()
    self._core_v1 = kubernetes.client.CoreV1Api()
    self._pod_restart_counts: dict = {}

  def detect_oom_kills(self) -> bool:
    crashed = False
    try:
      pods = self._core_v1.list_namespaced_pod(
        self._cfg.namespace, label_selector="app=target-app"
      )
      for pod in pods.items:
        if not pod.status or not pod.status.container_statuses:
          continue
        for cs in pod.status.container_statuses:
          if cs.name != "target-app":
            continue
          current_restarts = cs.restart_count
          pod_uid = pod.metadata.uid
          
          if pod_uid in self._pod_restart_counts:
            if current_restarts > self._pod_restart_counts[pod_uid]:
              logger.warning(
                "[%s] Pod %s restarted! (count: %d -> %d)",
                self._cfg.namespace, pod.metadata.name, 
                self._pod_restart_counts[pod_uid], current_restarts
              )
              crashed = True
              self._force_scale_up_v = True
              
              # Log termination reason if available
              if cs.last_state and cs.last_state.terminated:
                logger.warning(
                  "[%s] Pod %s termination reason: %s",
                  self._cfg.namespace, pod.metadata.name,
                  cs.last_state.terminated.reason
                )

          self._pod_restart_counts[pod_uid] = current_restarts
    except Exception as e:
      logger.error("Error checking pod status for OOM kills: %s", e)
      
    return crashed

  def reconcile(
    self,
    predicted_cpu_cores: float,
    actual_cpu_cores: float,
    predicted_ram_gb: float,
    actual_ram_gb: float,
    cycle_number: int = 0,
  ) -> ScalingEvent:
    cfg = self._cfg
    event = ScalingEvent(
      namespace=cfg.namespace,
      model_name=cfg.model_name,
      dataset_slug=cfg.dataset_slug,
      scaling_mode=cfg.scaling_mode,
      predicted_cpu_cores=predicted_cpu_cores,
      actual_cpu_cores=actual_cpu_cores,
      predicted_ram_gb=predicted_ram_gb,
      actual_ram_gb=actual_ram_gb,
    )

    current_replicas = self._get_current_replicas()
    current_memory_mb = self._get_current_memory_mb()

    event.current_replicas = current_replicas
    event.current_memory_mb = current_memory_mb

    event.sla_cpu_violation = (
      (actual_cpu_cores / max(current_replicas, 1)) > cfg.sla_cpu_threshold
    )
    event.sla_ram_violation = (
      (actual_ram_gb * 1024 / max(current_replicas, 1)) > cfg.sla_ram_threshold_mb
    )

    if cfg.scaling_mode in ("horizontal", "combined"):
      cur_r, des_r, action_h = self._reconcile_horizontal(
        predicted_cpu_cores, current_replicas
      )
      event.desired_replicas = des_r
      event.action_horizontal = action_h
    else:
      event.desired_replicas = current_replicas
      event.action_horizontal = "no_change"

    if cfg.scaling_mode in ("vertical", "combined"):
      cur_m, des_m, action_v = self._reconcile_vertical(
        predicted_ram_gb, current_memory_mb, current_replicas, cycle_number
      )
      event.desired_memory_mb = des_m
      event.action_vertical = action_v
    else:
      event.desired_memory_mb = current_memory_mb
      event.action_vertical = "no_change"

    try:
      if cfg.scaling_mode == "combined":
        needs_h = event.desired_replicas != current_replicas
        needs_v = event.desired_memory_mb != current_memory_mb
        if needs_h and needs_v:
          event.pod_restart_count = current_replicas
          self._patch_combined(event.desired_replicas, event.desired_memory_mb, event)
        elif needs_h:
          self._patch_replicas(event.desired_replicas, event)
        elif needs_v:
          event.pod_restart_count = current_replicas
          self._patch_memory(event.desired_memory_mb)
      elif cfg.scaling_mode == "horizontal":
        if event.desired_replicas != current_replicas:
          self._patch_replicas(event.desired_replicas, event)
      elif cfg.scaling_mode == "vertical":
        if event.desired_memory_mb != current_memory_mb:
          event.pod_restart_count = current_replicas
          self._patch_memory(event.desired_memory_mb)
    except Exception as exc:
      logger.error("Kubernetes patch failed: %s", exc)

    return event

  def _reconcile_horizontal(
    self, predicted_cpu_cores: float, current_replicas: int
  ) -> Tuple[int, int, str]:
    cfg = self._cfg
    desired = math.ceil(predicted_cpu_cores / cfg.cpu_target_per_replica)
    desired = max(cfg.min_replicas, min(cfg.max_replicas, desired))
    now = time.monotonic()

    if desired > current_replicas:
      if now - self._last_scale_up_h < cfg.scale_up_cooldown_seconds:
        return current_replicas, current_replicas, "cooldown"
      self._last_scale_up_h = now
      return current_replicas, desired, "scale_up"
    elif desired < current_replicas:
      if now - self._last_scale_down_h < cfg.scale_down_cooldown_seconds:
        return current_replicas, current_replicas, "cooldown"
      self._last_scale_down_h = now
      return current_replicas, desired, "scale_down"
    return current_replicas, current_replicas, "no_change"

  def _reconcile_vertical(
    self, predicted_ram_gb: float, current_memory_mb: int, current_replicas: int, cycle_number: int
  ) -> Tuple[int, int, str]:
    cfg = self._cfg
    desired = math.ceil((predicted_ram_gb * 1024) / max(current_replicas, 1))
    
    desired = math.ceil(desired / cfg.memory_step_mb) * cfg.memory_step_mb

    actual_max_per_pod = int(cfg.max_memory_mb / max(current_replicas, 1))
    desired = max(cfg.min_memory_mb, min(actual_max_per_pod, desired))
    now = time.monotonic()


    if getattr(self, "_force_scale_up_v", False):
      if desired > current_memory_mb:
        logger.warning("Bypassing vertical scale-up cooldown due to recent OOM kill!")
        self._force_scale_up_v = False
        self._last_scale_v = now
        return current_memory_mb, desired, "scale_up"
      else:
        self._force_scale_up_v = False

    if desired > current_memory_mb:
      if now - self._last_scale_v < cfg.vertical_cooldown_seconds:
        return current_memory_mb, current_memory_mb, "cooldown"
      self._last_scale_v = now
      return current_memory_mb, desired, "scale_up"
    elif desired < current_memory_mb:
      if now - self._last_scale_v < cfg.vertical_cooldown_seconds:
        return current_memory_mb, current_memory_mb, "cooldown"
      
      if now - self._start_time < 120.0:
        logger.info("Ignoring vertical scale-down during 120s startup grace period.")
        return current_memory_mb, current_memory_mb, "cooldown"

      max_step_down = current_memory_mb // 2
      bounded = max(desired, current_memory_mb - max_step_down)
      bounded = round(bounded / cfg.memory_step_mb) * cfg.memory_step_mb
      bounded = max(cfg.min_memory_mb, bounded)
      if bounded != desired:
        logger.info(
          "Vertical scale-down bounded: predicted %dMB capped to %dMB (50%% step limit from %dMB)",
          desired, bounded, current_memory_mb,
        )
      self._last_scale_v = now
      return current_memory_mb, bounded, "scale_down"

    return current_memory_mb, current_memory_mb, "no_change"

  def _patch_replicas(self, replicas: int, event: ScalingEvent = None) -> None:
    logger.info("Patching replicas → %d", replicas)
    start_time = time.monotonic()
    self._apps.patch_namespaced_deployment(
      name=self._cfg.deployment_name,
      namespace=self._cfg.namespace,
      body={"spec": {"replicas": replicas}},
    )
    if event and event.action_horizontal == "scale_up":
      import threading
      def _wait_for_ready():
        while True:
          try:
            pods = self._core_v1.list_namespaced_pod(
              namespace=self._cfg.namespace,
              label_selector=f"app={self._cfg.deployment_name}"
            )
            ready_count = 0
            for pod in pods.items:
              if pod.status.phase == "Running" and pod.status.conditions:
                for cond in pod.status.conditions:
                  if cond.type == "Ready" and cond.status == "True":
                    ready_count += 1
                    break
            if ready_count >= replicas:
              event.pod_ready_latency_seconds = round(time.monotonic() - start_time, 2)
              break
          except Exception as e:
            logger.debug("Failed to list pods during readiness wait: %s", e)
          time.sleep(1)
          if time.monotonic() - start_time > 300:  # 5 min timeout
            break
      threading.Thread(target=_wait_for_ready, daemon=True).start()

  def _patch_memory(self, memory_mb: int) -> None:
    limit = f"{memory_mb}Mi"
    request = "256Mi"  # Fixed low request to ensure pods can always be scheduled
    logger.info("Patching memory → limit=%s request=%s", limit, request)
    self._apps.patch_namespaced_deployment(
      name=self._cfg.deployment_name,
      namespace=self._cfg.namespace,
      body={
        "spec": {
          "template": {
            "spec": {
              "containers": [
                {
                  "name": self._cfg.deployment_name,
                  "resources": {
                    "limits": {"memory": limit},
                    "requests": {"memory": request},
                  },
                }
              ]
            }
          }
        }
      },
    )

  def _patch_combined(self, replicas: int, memory_mb: int, event: ScalingEvent = None) -> None:
    limit = f"{memory_mb}Mi"
    request = f"{int(memory_mb * 0.8)}Mi"
    logger.info("Patching combined: replicas=%d memory=%s", replicas, limit)
    start_time = time.monotonic()
    self._apps.patch_namespaced_deployment(
      name=self._cfg.deployment_name,
      namespace=self._cfg.namespace,
      body={
        "spec": {
          "replicas": replicas,
          "template": {
            "spec": {
              "containers": [
                {
                  "name": self._cfg.deployment_name,
                  "resources": {
                    "limits": {"memory": limit},
                    "requests": {"memory": request},
                  },
                }
              ]
            }
          },
        }
      },
    )
    if event and event.action_horizontal == "scale_up":
      import threading
      def _wait_for_ready():
        while True:
          try:
            pods = self._core_v1.list_namespaced_pod(
              namespace=self._cfg.namespace,
              label_selector=f"app={self._cfg.deployment_name}"
            )
            ready_count = 0
            for pod in pods.items:
              if pod.status.phase == "Running" and pod.status.conditions:
                for cond in pod.status.conditions:
                  if cond.type == "Ready" and cond.status == "True":
                    ready_count += 1
                    break
            if ready_count >= replicas:
              event.pod_ready_latency_seconds = round(time.monotonic() - start_time, 2)
              break
          except Exception as e:
            logger.debug("Failed to list pods during readiness wait: %s", e)
          time.sleep(1)
          if time.monotonic() - start_time > 300:  # 5 min timeout
            break
      threading.Thread(target=_wait_for_ready, daemon=True).start()

  def _get_current_replicas(self) -> int:
    try:
      dep = self._apps.read_namespaced_deployment(
        name=self._cfg.deployment_name,
        namespace=self._cfg.namespace,
      )
      return dep.spec.replicas or 1
    except Exception as exc:
      logger.error("Failed to read replicas: %s", exc)
      return 1

  def _get_current_memory_mb(self) -> int:
    try:
      dep = self._apps.read_namespaced_deployment(
        name=self._cfg.deployment_name,
        namespace=self._cfg.namespace,
      )
      containers = dep.spec.template.spec.containers
      for c in containers:
        if c.name == self._cfg.deployment_name:
          limits = c.resources.limits or {}
          mem = limits.get("memory", "0")
          return _parse_memory_quantity(mem)
      return 0
    except Exception as exc:
      logger.error("Failed to read memory limit: %s", exc)
      return 0
