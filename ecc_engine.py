import random
import json
import time
import base64

# --- PURE PYTHON "TINY CRYPTO" (No external libs needed) ---
# Implements basic RSA + XOR for demonstration purposes
# This guarantees compilation because it uses 0 dependencies.

class IoT_Crypto:
    def __init__(self):
        # Generate simple RSA keys (for demo speed)
        self.p = 61
        self.q = 53
        self.n = self.p * self.q
        self.phi = (self.p - 1) * (self.q - 1)
        self.e = 17
        self.d = pow(self.e, -1, self.phi)
        self.public_key = (self.e, self.n)
        self.private_key = (self.d, self.n)
        self.shared_secret = None

    def get_public_pem(self):
        # Send as string "e,n"
        return f"{self.e},{self.n}"

    def generate_challenge(self):
        nonce = str(random.getrandbits(64))
        timestamp = int(time.time())
        return nonce, timestamp

    def sign_data(self, data_str):
        # RSA Sign: s = m^d mod n
        # Hash data simple sum for demo speed
        h = sum(ord(c) for c in data_str)
        signature = pow(h, self.d, self.n)
        return str(signature)

    def verify_signature(self, public_str, data_str, signature_str):
        try:
            e, n = map(int, public_str.split(','))
            s = int(signature_str)
            # Verify: h = s^e mod n
            h_verify = pow(s, e, n)
            h_real = sum(ord(c) for c in data_str)
            # Modulo math check
            return (h_verify == h_real % n)
        except:
            return False

    def derive_shared_secret(self, peer_public_str):
        # Diffie-Hellman simulation using RSA keys
        # Session Key = Peer_N % 255 (Simple byte key)
        _, n = map(int, peer_public_str.split(','))
        self.shared_secret = n % 255
        return self.shared_secret

    def encrypt_medical_data(self, plaintext):
        if self.shared_secret is None: raise Exception("No Handshake")
        # XOR Cipher
        encrypted = []
        for char in plaintext:
            encrypted.append(chr(ord(char) ^ self.shared_secret))
        # Base64 encode safe transmission
        return base64.b64encode("".join(encrypted).encode()).decode()

    def decrypt_medical_data(self, b64_data):
        if self.shared_secret is None: raise Exception("No Handshake")
        encrypted_str = base64.b64decode(b64_data).decode()
        decrypted = []
        for char in encrypted_str:
            decrypted.append(chr(ord(char) ^ self.shared_secret))
        return "".join(decrypted)
