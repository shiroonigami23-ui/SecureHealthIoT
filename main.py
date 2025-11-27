from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.toolbar import MDTopAppBar
from kivy.clock import Clock
import ecc_engine  # Importing your crypto engine
import random
import json

class DoctorDashboard(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sensor = ecc_engine.IoT_Crypto() # The "Patient Device"
        self.doctor = ecc_engine.IoT_Crypto() # The "Doctor App"
        self.is_connected = False

        # --- LAYOUT ---
        layout = MDBoxLayout(orientation='vertical')
        
        # 1. Toolbar
        toolbar = MDTopAppBar(title="Secure Health Monitor")
        toolbar.elevation = 10
        layout.add_widget(toolbar)

        # 2. Status Card
        self.status_card = MDCard(
            size_hint=(0.9, None), height="100dp",
            pos_hint={"center_x": 0.5},
            padding="10dp",
            md_bg_color="#eeeeee"
        )
        self.status_label = MDLabel(
            text="STATUS: Disconnected",
            halign="center",
            theme_text_color="Error",
            font_style="H6"
        )
        self.status_card.add_widget(self.status_label)
        layout.add_widget(MDLabel(size_hint_y=None, height="20dp")) # Spacer
        layout.add_widget(self.status_card)

        # 3. Data Display Area
        self.data_label = MDLabel(
            text="--",
            halign="center",
            font_style="H2",
            theme_text_color="Primary"
        )
        layout.add_widget(self.data_label)

        # 4. Buttons
        btn_layout = MDBoxLayout(spacing="20dp", padding="20dp", size_hint_y=None, height="100dp")
        
        self.btn_connect = MDRaisedButton(
            text="HANDSHAKE & CONNECT",
            size_hint_x=1,
            md_bg_color="#1976D2",
            on_release=self.run_protocol
        )
        btn_layout.add_widget(self.btn_connect)
        layout.add_widget(btn_layout)

        # Add generic widget to fill space
        layout.add_widget(MDLabel())
        
        self.add_widget(layout)

    def run_protocol(self, instance):
        if self.is_connected:
            self.disconnect()
            return

        self.status_label.text = "Negotiating Keys..."
        self.status_label.theme_text_color = "Custom"
        self.status_label.text_color = "orange"
        
        # Schedule the heavy crypto work slightly later to show UI update
        Clock.schedule_once(self._execute_handshake, 0.5)

    def _execute_handshake(self, dt):
        try:
            # --- STEP 1: Exchange Keys ---
            sensor_pem = self.sensor.get_public_pem()
            doctor_pem = self.doctor.get_public_pem()

            # --- STEP 2: Verify Identity ---
            nonce, ts = self.doctor.generate_challenge()
            challenge = nonce + str(ts).encode()
            signature = self.sensor.sign_data(challenge)
            
            if self.doctor.verify_signature(sensor_pem, challenge, signature):
                # --- STEP 3: Establish Tunnel ---
                self.sensor.derive_shared_secret(doctor_pem)
                self.doctor.derive_shared_secret(sensor_pem)
                
                self.is_connected = True
                self.status_label.text = "SECURE TUNNEL ACTIVE"
                self.status_label.text_color = "green"
                self.btn_connect.text = "DISCONNECT"
                self.btn_connect.md_bg_color = "#d32f2f"
                
                # Start receiving data
                self.data_event = Clock.schedule_interval(self.update_vitals, 1)
            else:
                self.status_label.text = "AUTH FAILED: INTRUDER"
                self.status_label.text_color = "red"

        except Exception as e:
            self.status_label.text = f"Error: {str(e)}"

    def update_vitals(self, dt):
        # Simulate Sensor Data
        hr = random.randint(60, 100)
        spo2 = random.randint(95, 99)
        msg = f"HR: {hr} | SpO2: {spo2}%"
        
        # Encrypt (Sensor Side)
        encrypted = self.sensor.encrypt_medical_data(msg)
        
        # Decrypt (Doctor Side)
        decrypted = self.doctor.decrypt_medical_data(encrypted).decode()
        
        self.data_label.text = decrypted

    def disconnect(self):
        self.is_connected = False
        Clock.unschedule(self.data_event)
        self.status_label.text = "STATUS: Disconnected"
        self.status_label.text_color = "red"
        self.btn_connect.text = "HANDSHAKE & CONNECT"
        self.btn_connect.md_bg_color = "#1976D2"
        self.data_label.text = "--"

class HealthApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Blue"
        return DoctorDashboard()

if __name__ == "__main__":
    HealthApp().run()