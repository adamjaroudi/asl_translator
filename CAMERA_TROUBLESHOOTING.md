# Camera not working — settings checklist

If you see a black screen, "camera blocked" icon, or "Cannot open webcam", try these in order.

---

## 1. Windows camera privacy (most common)

**Windows 11**

1. Press **Win + I** to open **Settings**.
2. Go to **Privacy & security** → **Camera**.
3. Turn **Camera access** **On**.
4. Turn **Let desktop apps access your camera** **On**.
5. Under "Choose which apps can access your camera", make sure **Camera** (or your browser/app) is allowed if listed.

**Windows 10**

1. Press **Win + I** → **Privacy** → **Camera** (left side).
2. Turn **Allow apps to access your camera** **On**.
3. Turn **Allow desktop apps to access your camera** **On**.

Then **close and reopen** your terminal/Python script and try again.

---

## 2. Close other apps using the camera

Only one app can use the camera at a time. Close or minimize:

- **Zoom, Microsoft Teams, Skype, Discord** (video call apps)
- **Browser** (Chrome/Edge with a video call or site that asked for camera)
- **Windows Camera** app (if open)
- Any other Python script or app that might use the webcam

Then run your app again (e.g. `python asl_translator.py`).

---

## 3. Test in Windows Camera app

1. Open the **Camera** app (search "Camera" in the Start menu).
2. If the Camera app shows a black screen or an error, the problem is at Windows/driver level, not this project.
3. If the Camera app **does** show your face, then Windows allows the camera — close it and run your Python app again.

---

## 4. Allow Python/terminal through firewall (rare)

Some security software blocks camera access for unknown apps.

- **Windows Defender / Security**: Open **Windows Security** → **App & browser control** → **App privacy settings** → check **Camera** and allow desktop apps if needed.
- **Third‑party antivirus**: In its settings, look for "Camera" or "Webcam" and allow **Python** or **Command Prompt** / **Windows Terminal** or **Cursor**.

---

## 5. Check device manager (driver issues)

1. Press **Win + X** → **Device Manager**.
2. Expand **Cameras** or **Imaging devices**.
3. If your webcam has a yellow warning: right‑click → **Update driver** or **Uninstall device** (then restart so Windows reinstalls it).
4. If the camera is listed and has no warning, the driver is usually fine.

