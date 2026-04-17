"""Environment metadata collection for $startup events.

Collects platform, runtime, OS, framework, cloud provider, and CI
environment details. Used by the client to populate the $startup
event with debugging and segmentation data.
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import sys

log = logging.getLogger("litmus")


def _detect_framework() -> str | None:
    """Check sys.modules for common Python web frameworks."""
    frameworks = [
        "fastapi",
        "django",
        "flask",
        "starlette",
        "sanic",
        "litestar",
        "falcon",
        "tornado",
        "aiohttp",
        "quart",
        "bottle",
        "pyramid",
        "chalice",
    ]
    for name in frameworks:
        if name in sys.modules:
            return name
    return None


def _detect_cloud() -> dict | None:
    """Detect cloud/serverless provider from environment variables."""
    env = os.environ
    if env.get("VERCEL"):
        return {"cloud_provider": "vercel", "cloud_region": env.get("VERCEL_REGION")}
    if env.get("AWS_REGION"):
        return {
            "cloud_provider": "aws",
            "cloud_region": env.get("AWS_REGION"),
            "cloud_platform": env.get("AWS_EXECUTION_ENV"),
        }
    if env.get("FLY_REGION"):
        return {"cloud_provider": "fly", "cloud_region": env.get("FLY_REGION")}
    if env.get("RAILWAY_ENVIRONMENT_NAME"):
        return {
            "cloud_provider": "railway",
            "cloud_environment": env.get("RAILWAY_ENVIRONMENT_NAME"),
        }
    if env.get("RENDER"):
        return {"cloud_provider": "render", "cloud_region": env.get("RENDER_REGION")}
    if env.get("GCP_PROJECT") or env.get("GOOGLE_CLOUD_PROJECT"):
        return {"cloud_provider": "gcp"}
    if env.get("WEBSITE_SITE_NAME") and env.get("REGION_NAME"):
        return {"cloud_provider": "azure", "cloud_region": env.get("REGION_NAME")}
    if env.get("DYNO"):
        return {"cloud_provider": "heroku"}
    if env.get("NETLIFY"):
        return {"cloud_provider": "netlify"}
    return None


def _detect_ci() -> str | None:
    """Detect CI environment from env vars."""
    env = os.environ
    if env.get("GITHUB_ACTIONS"):
        return "github_actions"
    if env.get("GITLAB_CI"):
        return "gitlab_ci"
    if env.get("CIRCLECI"):
        return "circleci"
    if env.get("BUILDKITE"):
        return "buildkite"
    if env.get("JENKINS_URL"):
        return "jenkins"
    if env.get("CODEBUILD_BUILD_ID"):
        return "aws_codebuild"
    if env.get("CI"):
        return "unknown_ci"
    return None


def _hostname() -> str | None:
    """Best-effort raw hostname.

    socket.gethostname() can fail in weird sandboxes (missing /etc/hostname,
    restricted seccomp) or return an empty string. Swallow failures so we
    never block $startup on metadata collection — hostname is nice to have,
    never required.
    """
    try:
        name = socket.gethostname()
    except OSError:
        log.debug("socket.gethostname() failed", exc_info=True)
        return None
    return name or None


def collect_startup_metadata() -> dict:
    """Collect all environment metadata for the $startup event."""
    meta: dict = {
        "platform": sys.platform,
        "python_version": platform.python_version(),
        "runtime": platform.python_implementation().lower(),
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
    }

    # Raw hostname. For containerized deploys this is often the pod/container
    # name (Railway, Fly, Kubernetes) which is exactly what you want for
    # "which instance is sending this?" debugging. Kept separate from the
    # derived cloud_provider field so ops can group by either.
    host = _hostname()
    if host:
        meta["hostname"] = host

    # CPU count
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        meta["cpu_count"] = cpu_count

    # Framework detection
    framework = _detect_framework()
    if framework:
        meta["framework"] = framework

    # Cloud provider
    cloud = _detect_cloud()
    if cloud:
        meta.update(cloud)

    # CI environment
    ci = _detect_ci()
    if ci:
        meta["ci"] = ci

    return meta
