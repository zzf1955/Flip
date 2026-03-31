"""ComfyUI REST API client for workflow automation."""
import json
import mimetypes
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path


class ComfyUIClient:
    def __init__(self, host="127.0.0.1", port=8000):
        self.base = f"http://{host}:{port}"

    def upload_image(self, filepath, subfolder="", overwrite=True):
        """Upload an image to ComfyUI input directory."""
        filepath = Path(filepath)
        content_type = mimetypes.guess_type(str(filepath))[0] or "image/png"

        boundary = f"----ComfyBoundary{random.randint(100000, 999999)}"
        body = b""

        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="image"; filename="{filepath.name}"\r\n'.encode()
        body += f"Content-Type: {content_type}\r\n\r\n".encode()
        body += filepath.read_bytes()
        body += b"\r\n"

        body += f"--{boundary}\r\n".encode()
        body += b'Content-Disposition: form-data; name="type"\r\n\r\ninput\r\n'

        body += f"--{boundary}\r\n".encode()
        body += b'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        body += f"{'true' if overwrite else 'false'}\r\n".encode()

        if subfolder:
            body += f"--{boundary}\r\n".encode()
            body += b'Content-Disposition: form-data; name="subfolder"\r\n\r\n'
            body += f"{subfolder}\r\n".encode()

        body += f"--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            f"{self.base}/api/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        print(f"  Uploaded: {result.get('name', filepath.name)}")
        return result

    def queue_prompt(self, workflow):
        """Submit a workflow for execution."""
        payload = json.dumps({"prompt": workflow}).encode()
        req = urllib.request.Request(
            f"{self.base}/api/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        prompt_id = result["prompt_id"]
        print(f"  Queued: {prompt_id}")
        return prompt_id

    def wait_for_completion(self, prompt_id, timeout=600, interval=2):
        """Poll until the prompt completes."""
        start = time.time()
        while time.time() - start < timeout:
            url = f"{self.base}/api/history/{prompt_id}"
            resp = urllib.request.urlopen(url)
            history = json.loads(resp.read())
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("completed", False) or status.get("status_str") == "success":
                    print(f"  Completed in {time.time() - start:.1f}s")
                    return entry
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"Prompt failed: {msgs}")
            time.sleep(interval)
        raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout}s")

    def get_output_images(self, history_entry):
        """Extract output image info from history entry."""
        images = []
        for node_id, node_output in history_entry.get("outputs", {}).items():
            if "images" in node_output:
                for img in node_output["images"]:
                    images.append(img)
        return images

    def download_image(self, filename, subfolder="", folder_type="output"):
        """Download an output image."""
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        })
        url = f"{self.base}/api/view?{params}"
        resp = urllib.request.urlopen(url)
        return resp.read()

    def download_results(self, history_entry, output_dir=None):
        """Download all output images from a completed prompt."""
        output_images = self.get_output_images(history_entry)
        if not output_images:
            print("  WARNING: No output images found")
            return []

        results = []
        for img_info in output_images:
            filename = img_info["filename"]
            subfolder = img_info.get("subfolder", "")
            data = self.download_image(filename, subfolder)

            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / filename
                save_path.write_bytes(data)
                print(f"  Saved: {save_path}")
                results.append(str(save_path))
            else:
                print(f"  Output: {filename} ({len(data)} bytes)")
                results.append(filename)

        return results
