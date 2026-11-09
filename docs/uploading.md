# Uploading

The uploader (`src/data_uploader/uploader.py`, class `DataUploader`) sends finished events from `data/pending_upload/` to the FaunaScope server over HTTPS. It is built for a cellular link that drops, a data plan that is limited, and a server that may be unreachable for days: no event is ever deleted, an interrupted video resumes where it stopped, and an event leaves the pending folder only after the server has confirmed it.

## Credentials

The device holds one secret, its upload token, and knows one address, the server's URL. Both are in `.env`:

```sh
SERVER_URL=https://faunascope.txst.edu
DEVICE_TOKEN=<the token>
UPLOADER_ENABLED=true
```

The server's administrator registers the device under its `DEVICE_ID` and gives out the token once; the server stores only a hash of it. The token admits uploads for that one device ID only: the server refuses metadata that names another device and events another device has already sent. To check a token, ask the server about an event it has never seen. A valid token gets `{"offset":0,"received":false}`; an invalid one gets `401`:

```sh
curl -H "Authorization: Bearer $DEVICE_TOKEN" $SERVER_URL/api/v1/events/event_20000101-000000_00000000
```

If `SERVER_URL` or `DEVICE_TOKEN` is missing, the uploader logs an error and stays idle, and recording continues.

## The protocol

Every request goes to `<SERVER_URL>/api/v1/events/<event_id>` with the header `Authorization: Bearer <token>`.

| Step | Request | The server answers |
|---|---|---|
| 1. Ask | `GET /api/v1/events/<event_id>` | `{"received": true}` if it already holds the event, otherwise `{"received": false, "offset": n}`, the number of bytes of the video it holds |
| 2. Send the video | `PATCH /api/v1/events/<event_id>/video`, header `Upload-Offset: n`, body: the next chunk (`chunk_size_kb`, 1 MB; the server accepts up to 2 MB) | `200 {"offset": n + length}`, or `409 {"offset": m}` if it holds a different number of bytes, in which case the device continues from `m` |
| 3. Finish | `POST /api/v1/events/<event_id>` with `{"metadata": <the JSON>, "video_size": N, "video_sha256": "<hex>"}`, gzipped (`Content-Encoding: gzip`) | `201` after it has checked the size, checksum, and playability of the video and stored both files; `200` if it already held the event; `409 {"offset": m}` if the video is incomplete; `422` if the video does not match |

Only after `200` or `201` does the uploader move the event's two files to `data/uploaded/`. If a connection drops in the middle of a chunk, the next attempt asks again and resumes from the server's offset, so at most one chunk is sent twice. If the server's `201` is lost on the way back, the repeated `POST` gets `200`. If the move to `uploaded/` fails, the files stay in the pending folder and the next pass finds that the server already holds the event.

The metadata is sent gzipped because it can be large: in the archive, half the events have less than 0.2 MB of metadata, but an event with many tracks reaches 7 MB, beyond the server's 2 MB limit per request. Compressed, the largest is 0.17 MB. The server also limits each device to a daily volume (2 GB by default) and each address to 10 requests a second; the uploader leaves at least 0.2 seconds between requests, so even on a fast link it stays under that limit.

## Retries

The uploader distinguishes failures of the link from failures of a particular event, because they call for different responses.

- **Link failures**: no connection, a timeout, HTTP 5xx, `429`, or `401` (which can follow a token change on the server). Nothing is wrong with the event, so nothing counts against it. The current pass stops and uploads pause for 1, 2, 4, ... minutes, up to an hour; the next pass resumes from the server's offset. A failed attempt costs at most one chunk and a TLS handshake.
- **Event failures**: any other refusal (`400`, `413`, `422`, ...) or local files that cannot be read. These count against the event. After `event_attempts` (3) failures it is deferred for `deferral_hours` (24); after `max_deferrals` (3) deferrals it gets a marker file, `<event_id>.json.failed`, which contains the time and the last error. Its files are never deleted. Deleting the marker queues the event again.

## The daily budget

`daily_upload_limit_mb` (800) caps the video bytes sent per calendar day. Each chunk is counted before it is sent; when the next chunk would exceed the budget, uploads stop until the next day. 800 MB a day suits a 30 GB monthly plan and leaves room for the live view and system updates, and at recent file sizes it is about 25 to 30 events a day, so new events keep up while a backlog drains slowly.

## Order and state

Each pass, every `scan_interval_seconds` (60), handles the pending events oldest first (event names begin with the time), skipping events with a `.failed` marker or an unexpired deferral, and stops at the first link failure or when the budget is spent. Attempt counts, deferrals, and the bytes sent today are kept in `data/pending_upload/.upload_state.json`, written atomically, so a restart neither resets the limits nor forgets them.

Events in `data/held_blank/` are never uploaded; moving a held event's files into `data/pending_upload/` queues it.

## Settings

All in the `uploader` section of `config/config.yaml`; the address and token come from `.env`.

| Setting | Default | Meaning |
|---|---|---|
| `enabled` | `${UPLOADER_ENABLED}` | Run the uploader at all |
| `scan_interval_seconds` | 60 | Time between passes |
| `server_url`, `device_token` | `${SERVER_URL}`, `${DEVICE_TOKEN}` | The server and this device's token |
| `chunk_size_kb` | 1024 | Size of each video chunk; smaller chunks waste less on a poor link |
| `daily_upload_limit_mb` | 800 | Video data sent per calendar day |
| `event_attempts`, `deferral_hours`, `max_deferrals` | 3, 24, 3 | Retries for rejected events |
| `request_timeout_seconds` | 60 | Timeout of each request |
