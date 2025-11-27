[app]
title = SecureHealth
package.name = securehealth
package.domain = org.iot
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1
# CRITICAL: ONLY PURE PYTHON LIBRARIES HERE.
# removed: cryptography, cffi, pycparser, ecdsa
# We will bundle a local pure-python crypto inside the app instead of installing it.
requirements = python3,kivy==2.2.0,kivymd==1.1.1
orientation = portrait
fullscreen = 0
android.permissions = INTERNET
android.api = 33
android.minapi = 21
android.ndk = 25b
android.accept_sdk_license = True
p4a.branch = master

[buildozer]
log_level = 2
warn_on_root = 1
