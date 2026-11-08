### **Specification for Module C: Data Uploader**

#### **1. Overview**

This document outlines the functional requirements and technical specifications for the Data Uploader module (Module C). This module operates as a resilient, background service responsible for the reliable transmission of processed data from the edge device to a remote cloud server. Its primary function is to periodically scan for completed event packages, send them to the FaunaScope server over HTTPS, and handle network failures gracefully to ensure no data is lost.

#### **2. Core Functional Requirements**

*   **Asynchronous Operation:** The module must run independently of the data capture and analysis modules, operating on a periodic schedule.
*   **Filesystem-Based Queueing:** The module will use a designated local directory as its input queue, processing data packages as they appear.
*   **Transactional Uploads:** Each event package (composed of a video and a JSON file) shall be treated as a single, atomic unit of work. The entire upload process for a single event must succeed before local files are cleaned up.
*   **Secure Data Transfer:** All transfers go to the server's upload API over HTTPS, authenticated by the device's token. The device holds no database or SSH credentials; the server creates the database records itself.
*   **Resumable Transfer:** Videos are sent in chunks, and an interrupted upload resumes at the last chunk the server confirmed.
*   **Resilience and Auto-Retry:** In the event of a network or server error at any stage of the upload process, the local files for the current job must remain untouched, allowing the system to automatically retry the upload later. Network failures and rejected events are handled differently (see section 5).
*   **Data Budget:** The bytes sent per calendar day are capped, so that a backlog cannot use up a cellular data plan.

#### **3. Interface Contract: Communication with Module B**

The interface for this module is based on the local filesystem, providing a decoupled "hand-off" point from the Animal Analyzer (Module B).

*   **Input Interface:**
    *   **Location:** A designated "pending upload" directory (e.g., `data/pending_upload/`).
    *   **Trigger:** The presence of a `.json` file in this directory signals a new job.
    *   **Contract:** For every `[event_id].json` file, a corresponding `[event_id].mp4` file is expected to exist in the same directory.
*   **Output Interface:**
    *   **Location:** A designated "uploaded" directory (e.g., `data/uploaded/`).
    *   **Action:** Upon the successful completion of an entire upload transaction, the corresponding local `.json` and `.mp4` files are moved from the input directory to this output directory for archival purposes.

#### **4. Technical Implementation Specifications**

*   **Primary Libraries:** `requests`, `hashlib`, `os`, `time`, `json`, `shutil`.
*   **Class Structure:** All logic should be encapsulated within a Python class named `DataUploader`.
*   **Configuration Management:** All external parameters will be provided via a configuration dictionary.

    **Required Configuration Parameters (`uploader` section):**
    *   `scan_interval_seconds` (int): The number of seconds the module sleeps between upload passes.
    *   `server_url` (str): The server's address, e.g. `https://faunascope.txst.edu`.
    *   `device_token` (str): The device's upload token, issued by the server's administrator.
    *   `chunk_size_kb` (int): The size of the chunks a video is sent in (the server accepts up to 2048).
    *   `daily_upload_limit_mb` (int): The most video data to send per calendar day.
    *   `event_attempts` (int), `deferral_hours` (number), `max_deferrals` (int): How rejected events are retried.
    *   `request_timeout_seconds` (number): The timeout for each HTTP request.

#### **5. Processing Pipeline Logic**

1.  **Main Loop:** Every `scan_interval_seconds`, unless uploads are paused after a network failure, the uploader makes one pass over the `event_*.json` files in the pending directory, oldest first, skipping events with a `.failed` marker or a deferral that has not expired. A pass stops at the first network failure or when the day's budget is spent.

2.  **Single Event:** For an event `<event_id>`, with requests to `<server_url>/api/v1/events/<event_id>`:

    a. **Read Local Data:** Load the JSON file and find `<event_id>.mp4` next to it.
    b. **Ask the Server:** `GET` returns `{"received": true}` if the server already holds the event, which goes straight to cleanup, or the number of bytes of the video it holds (`offset`).
    c. **Send the Video:** From that offset, `PATCH .../video` sends one chunk at a time with an `Upload-Offset` header. Each chunk's size is counted against the day's budget before it is sent. If the server answers `409` with another offset, sending continues from that offset.
    d. **Finish:** `POST` sends `{"metadata": ..., "video_size": ..., "video_sha256": ...}`. The server answers `201` after verifying the video and storing both files where its worker imports them, or `200` if it already held the event.
    e. **Local Cleanup:** **If and only if the server answered `200` or `201`**, move the local `.json` and `.mp4` files to the `uploaded` directory.

3.  **Failures:**

    *   **Network failures** (no connection, timeout, HTTP 5xx, `429`, or `401`) pause all uploads for 1, 2, 4, ... minutes, up to 60, and do not count against the event. The next pass resumes from the server's offset.
    *   **Event failures** (any other refusal by the server, or unreadable local files) count against the event. After `event_attempts` failures it is deferred for `deferral_hours`; after `max_deferrals` deferrals it gets a `<event_id>.json.failed` marker. Its files are never deleted, and deleting the marker queues it again.
    *   Attempt counts, deferrals, and the bytes sent today are kept in `.upload_state.json` in the pending directory, so a restart neither resets the limits nor forgets them.
