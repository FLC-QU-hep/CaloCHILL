"""Workarounds for torch._dynamo bugs that abort a compile from its logging path."""

import collections
import functools
import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_pion_clouds_safe_no_tensor_aliasing_reason"


def patch_no_tensor_aliasing_reason() -> bool:
    """Stop a failed guard-source eval from killing the whole recompilation.

    When a frame is recompiled, dynamo first formats a human-readable "recompiled
    because ..." message. For a no-tensor-aliasing ("Duplicate tensor found") guard
    that means re-eval()ing the guard source expressions of the *previously* compiled
    variant (e.g. "L['masks'][3]") against the locals of the frame being recompiled --
    torch 2.10's recompilation_reason_for_no_tensor_aliasing_guard(), guards.py:4309.
    A source that no longer resolves there raises (IndexError: tuple index out of
    range for us), and because this runs inside _compile() the exception aborts the
    recompilation itself: dynamo logs "WON'T CONVERT <fn> ... due to: <traceback>" and
    the frame falls back to eager. We hit it on Compressor.forward/.encoder, whose
    `masks` list carries BlockMasks built from shared pad tensors.

    Replace the function with one that skips the sources it cannot resolve, so the
    message degrades instead of the compile. Diagnostics only -- no effect on the
    guards themselves or on the compiled code.

    Idempotent. Returns True if the patch is in place.
    """
    try:
        from torch._dynamo import guards as dynamo_guards
    except Exception:  # pragma: no cover - torch always importable in practice
        logger.debug("torch._dynamo.guards unavailable, no patch applied")
        return False

    original = getattr(
        dynamo_guards, "recompilation_reason_for_no_tensor_aliasing_guard", None
    )
    if original is None:
        # Upstream renamed or removed it: nothing to protect.
        return False
    if getattr(original, _PATCH_FLAG, False):
        return True

    @functools.wraps(original)
    def safe_recompilation_reason(guard_manager, scope):
        try:
            global_scope = dict(guard_manager.global_scope)
            ids_to_source = collections.defaultdict(list)
            unresolved = 0
            for tensor_source in guard_manager.no_tensor_aliasing_sources:
                global_scope["__compile_source__"] = tensor_source
                try:
                    tensor_id = id(eval(tensor_source, global_scope, scope))
                except Exception:
                    # Stale source for this frame -- the only thing upstream does
                    # not tolerate. It costs us one entry in the message.
                    unresolved += 1
                    continue
                ids_to_source[tensor_id].append(tensor_source)

            duplicate_tensors = [
                f"{sources}" for sources in ids_to_source.values() if len(sources) > 1
            ]
            reason = ", ".join(duplicate_tensors)
            if unresolved:
                reason += (
                    f" ({unresolved} guard source(s) do not resolve in this frame)"
                )
            return [f"Duplicate tensors found: {reason}"]
        except Exception:
            logger.debug("could not build a no-tensor-aliasing reason", exc_info=True)
            return ["Duplicate tensors found: <reason unavailable>"]

    setattr(safe_recompilation_reason, _PATCH_FLAG, True)
    dynamo_guards.recompilation_reason_for_no_tensor_aliasing_guard = (
        safe_recompilation_reason
    )
    return True
