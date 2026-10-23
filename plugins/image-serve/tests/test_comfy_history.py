"""A prompt ComfyUI gave up on is reported at once with its reason (comfy.history_error), not after a 60 s wait."""
from image_serve.comfy import history_error


def test_error_entry_gives_node_and_message_without_traceback():
    entry = {"status": {"status_str": "error", "completed": False, "messages": [
        ["execution_start", {"prompt_id": "p"}],
        ["execution_error", {"node_type": "KSampler", "exception_type": "RuntimeError",
                             "exception_message": "Failed to find C compiler.\n  Please specify via CC",
                             "traceback": ["File \"/home/someone/ComfyUI/x.py\""]}]]}, "outputs": {}}
    why = history_error(entry)
    assert why == "ComfyUI failed in KSampler: Failed to find C compiler. Please specify via CC"
    assert "/home/" not in why


def test_success_or_running_is_no_error():
    assert history_error({"status": {"status_str": "success"}, "outputs": {"9": {"images": []}}}) is None
    assert history_error({"outputs": {}}) is None and history_error(None) is None


def test_error_without_details_still_says_so():
    assert "ComfyUI failed" in history_error({"status": {"status_str": "error", "messages": []}})
