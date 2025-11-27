from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
import time
import json
import base64

# --- LIGHTWEIGHT ECC ENGINE ---

class IoT_Crypto:
    def __init__(self):
        # Generate ECC Keys (Curve SECP256R1 is standard for IoT)
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()
        self.shared_key = None # Will be established after handshake

    def get_public_pem(self):
        """Export Public Key to send to the other device"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

    def generate_challenge(self):
        """Create a random puzzle (Nonce) + Timestamp to stop Replay Attacks"""
        nonce = os.urandom(16)
        timestamp = int(time.time())
        return nonce, timestamp

    def sign_data(self, data_bytes):
        """Sign data with Private Key (Proof of Identity)"""
        signature = self.private_key.sign(
            data_bytes,
            ec.ECDSA(hashes.SHA256())
        )
        return signature

    def verify_signature(self, public_pem, data_bytes, signature):
        """Verify the other person is who they say they are"""
        try:
            # Load the other person's key
            peer_public_key = serialization.load_pem_public_key(public_pem.encode('utf-8'))
            peer_public_key.verify(
                signature,
                data_bytes,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except Exception as e:
            print(f"Verification Failed: {e}")
            return False

    def derive_shared_secret(self, peer_public_pem):
        """ECDH: Create a secret encryption key that only we two know"""
        peer_public_key = serialization.load_pem_public_key(peer_public_pem.encode('utf-8'))
        shared_secret = self.private_key.exchange(ec.ECDH(), peer_public_key)
        
        # Turn that raw math into a clean AES Key
        self.shared_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'handshake data',
        ).derive(shared_secret)
        
        return self.shared_key

    def encrypt_medical_data(self, plaintext):
        """Encrypts vitals (e.g., 'HeartRate: 72') using the Shared Key"""
        if not self.shared_key: raise Exception("Handshake not complete!")
        
        iv = os.urandom(12) # Initialization Vector
        # Use AES-GCM (Authenticated Encryption) - Very secure
        encryptor = Cipher(
            algorithms.AES(self.shared_key),
            modes.GCM(iv),
        ).encryptor()
        
        ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
        
        # Package it all up
        package = {
            "iv": base64.b64encode(iv).decode('utf-8'),
            "tag": base64.b64encode(encryptor.tag).decode('utf-8'),
            "data": base64.b64encode(ciphertext).decode('utf-8')
        }
        return json.dumps(package)

    def decrypt_medical_data(self, json_package):
        """Decrypts vitals"""
        if not self.shared_key: raise Exception("Handshake not complete!")
        
        pkg = json.loads(json_package)
        iv = base64.b64decode(pkg['iv'])
        tag = base64.b64decode(pkg['tag'])
        ciphertext = base64.b64decode(pkg['data'])
        
        decryptor = Cipher(
            algorithms.AES(self.shared_key),
            modes.GCM(iv, tag),
        ).decryptor()
        
        return decryptor.update(ciphertext) + decryptor.finalize()

# Test the Engine immediately when run
if __name__ == "__main__":
    print("Initializing IoT Sensor...")
    sensor = IoT_Crypto()
    
    print("Initializing Doctor App...")
    doctor = IoT_Crypto()
    
    print("\n--- STEP 1: KEY EXCHANGE ---")
    sensor_pem = sensor.get_public_pem()
    doctor_pem = doctor.get_public_pem()
    print("Keys exchanged successfully.")
    
    print("\n--- STEP 2: MUTUAL AUTHENTICATION ---")
    # Doctor challenges Sensor
    nonce, ts = doctor.generate_challenge()
    challenge_data = nonce + str(ts).encode()
    
    # Sensor signs the challenge
    signature = sensor.sign_data(challenge_data)
    
    # Doctor verifies
    is_valid = doctor.verify_signature(sensor_pem, challenge_data, signature)
    if is_valid:
        print("✅ MUTUAL AUTH SUCCESS: Sensor is legitimate.")
        
        # Derive encryption keys (ECDH)
        sensor.derive_shared_secret(doctor_pem)
        doctor.derive_shared_secret(sensor_pem)
        print("🔒 Secure Tunnel Established.")
        
        print("\n--- STEP 3: DATA TRANSMISSION ---")
        msg = "HeartRate: 88 BPM, SPO2: 98%"
        encrypted = sensor.encrypt_medical_data(msg)
        print(f"Encrypted Packet: {encrypted}")
        
        decrypted = doctor.decrypt_medical_data(encrypted)
        print(f"Decrypted Vitals: {decrypted.decode()}")
        
    else:
        print("❌ AUTH FAILED: Potential Intruder!")