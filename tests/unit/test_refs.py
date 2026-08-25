"""产物引用与请求指纹的领域规则。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.capability import CapabilityRequest
from app.domain.refs import ArtifactRef, ArtifactType


def test_version_is_mandatory() -> None:
    """设计文档 8.3：禁止隐式读取最新版参与正式生产。"""
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate({"type": "hot_video", "id": "hv_001"})


def test_version_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(type=ArtifactType.HOT_VIDEO, id="hv_001", version=0)


def test_fingerprint_is_stable_regardless_of_key_order() -> None:
    a = CapabilityRequest(parameters={"x": 1, "y": 2}, idempotency_key="same-key-1")
    b = CapabilityRequest(parameters={"y": 2, "x": 1}, idempotency_key="same-key-1")
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_when_inputs_change() -> None:
    """幂等键相同而指纹不同，说明调用方复用了幂等键——T1-4 要靠这个发现问题。"""
    base = CapabilityRequest(parameters={"x": 1}, idempotency_key="same-key-1")
    other = CapabilityRequest(parameters={"x": 2}, idempotency_key="same-key-1")
    assert base.fingerprint() != other.fingerprint()


def test_input_ref_version_participates_in_fingerprint() -> None:
    v1 = CapabilityRequest(
        input_refs=(ArtifactRef(type=ArtifactType.HOT_VIDEO, id="hv_001", version=1),),
        idempotency_key="same-key-1",
    )
    v2 = CapabilityRequest(
        input_refs=(ArtifactRef(type=ArtifactType.HOT_VIDEO, id="hv_001", version=2),),
        idempotency_key="same-key-1",
    )
    assert v1.fingerprint() != v2.fingerprint()
