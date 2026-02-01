"""
MEDİAPIPE EL ANALİZİ - GELİŞMİŞ
================================
+ Hareket hızı tespiti
+ Tehdit anında otomatik video kaydı
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
import time
import os
import math

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Kayıt klasörü
RECORD_DIR = "security_records"
if not os.path.exists(RECORD_DIR):
    os.makedirs(RECORD_DIR)


class MotionTracker:
    """Hareket hızı takibi"""

    def __init__(self):
        self.positions = deque(maxlen=15)
        self.velocities = deque(maxlen=10)

    def update(self, center):
        """Pozisyon güncelle ve hız hesapla"""
        self.positions.append((center, time.time()))

        if len(self.positions) >= 2:
            p1, t1 = self.positions[-2]
            p2, t2 = self.positions[-1]

            dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            dt = t2 - t1 if t2 - t1 > 0 else 0.001

            velocity = dist / dt
            self.velocities.append(velocity)

        return self.get_speed()

    def get_speed(self):
        """Ortalama hız"""
        if not self.velocities:
            return 0
        return sum(self.velocities) / len(self.velocities)

    def get_motion_level(self):
        """Hareket seviyesi"""
        speed = self.get_speed()

        if speed > 800:
            return "COK_HIZLI", 35, (0, 0, 255)
        elif speed > 400:
            return "HIZLI", 20, (0, 100, 255)
        elif speed > 150:
            return "NORMAL", 5, (0, 200, 255)
        else:
            return "YAVAS", 0, (0, 255, 0)


class ThreatRecorder:
    """Tehdit anında video kaydı"""

    def __init__(self, frame_size, fps=20):
        self.frame_size = frame_size
        self.fps = fps
        self.writer = None
        self.recording = False
        self.record_start = 0
        self.record_duration = 5  # 5 saniye kayıt
        self.buffer = deque(maxlen=fps * 2)  # 2 saniyelik ön buffer
        self.last_record_time = 0
        self.cooldown = 10  # 10 saniye bekleme

    def add_frame(self, frame):
        """Frame'i buffer'a ekle"""
        self.buffer.append(frame.copy())

    def start_recording(self):
        """Kaydı başlat"""
        if time.time() - self.last_record_time < self.cooldown:
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RECORD_DIR, f"threat_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fourcc, self.fps, self.frame_size)

        # Ön buffer'ı yaz
        for buffered_frame in self.buffer:
            if buffered_frame.shape[1] == self.frame_size[0]:
                self.writer.write(buffered_frame)

        self.recording = True
        self.record_start = time.time()
        self.last_record_time = time.time()

        print(f"\n🔴 KAYIT BAŞLADI: {filename}")
        return True

    def update(self, frame):
        """Kayıt güncelle"""
        self.add_frame(frame)

        if self.recording:
            if frame.shape[1] == self.frame_size[0]:
                self.writer.write(frame)

            elapsed = time.time() - self.record_start
            if elapsed >= self.record_duration:
                self.stop_recording()
                return False
            return True
        return False

    def stop_recording(self):
        """Kaydı durdur"""
        if self.writer:
            self.writer.release()
            self.writer = None
        self.recording = False
        print("⬛ KAYIT TAMAMLANDI\n")

    def is_recording(self):
        return self.recording


def count_fingers(hand_landmarks):
    """Parmak sayma"""
    tips = [4, 8, 12, 16, 20]
    pip = [3, 6, 10, 14, 18]

    fingers = 0

    # Başparmak
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers += 1

    # Diğer parmaklar
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pip[i]].y:
            fingers += 1

    return fingers


def get_gesture(fingers):
    """Hareket sınıflandırma"""
    if fingers == 0:
        return "YUMRUK", 70, (0, 0, 255)
    elif fingers == 5:
        return "ACIK_EL", 5, (0, 255, 0)
    elif fingers == 1:
        return "ISARET", 40, (0, 165, 255)
    elif fingers == 2:
        return "ZAFER", 10, (0, 255, 150)
    elif fingers == 3:
        return "UC", 20, (255, 255, 0)
    elif fingers == 4:
        return "DORT", 10, (200, 255, 0)
    return "BELIRSIZ", 25, (150, 150, 150)


def get_intent(gesture, threat):
    """Niyet belirleme"""
    if threat >= 80:
        return "TEHLIKELI!"
    elif gesture == "YUMRUK":
        return "SALDIRGAN"
    elif gesture == "ACIK_EL":
        return "DOST"
    elif gesture == "ISARET":
        return "UYARI"
    elif gesture == "ZAFER":
        return "BARIS"
    return "BELIRSIZ"


def main():
    print("=" * 55)
    print("   MEDİAPIPE EL ANALİZİ - GELİŞMİŞ")
    print("=" * 55)
    print()
    print("  Özellikler:")
    print("    ✓ Hareket hızı tespiti")
    print("    ✓ Tehdit anında otomatik video kaydı")
    print()
    print("  Hareketler:")
    print("    ✊ YUMRUK    → Saldırgan")
    print("    🖐 AÇIK EL   → Dost")
    print("    👆 İŞARET    → Uyarı")
    print("    ✌ ZAFER     → Barış")
    print()
    print(f"  Kayıtlar: {RECORD_DIR}/")
    print("  'q' = Çıkış")
    print("=" * 55)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Frame boyutunu al
    ret, test_frame = cap.read()
    if ret:
        h, w = test_frame.shape[:2]
        frame_size = (w, h)
    else:
        frame_size = (640, 480)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # Trackers
    motion_tracker = MotionTracker()
    recorder = ThreatRecorder(frame_size)

    events = deque(maxlen=5)
    last_alert = 0
    frame_count = 0

    print("\nKamera başlatılıyor...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "---"
        base_threat = 0
        motion_threat = 0
        total_threat = 0
        intent = "BEKLENIYOR"
        color = (150, 150, 150)
        fingers = 0
        speed = 0
        motion_level = "---"
        motion_color = (150, 150, 150)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

                # Parmak sayısı
                fingers = count_fingers(hand_landmarks)
                gesture, base_threat, color = get_gesture(fingers)

                # El merkezi
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                cx = int(sum(x_list) / len(x_list) * w)
                cy = int(sum(y_list) / len(y_list) * h)

                # Hareket hızı
                speed = motion_tracker.update((cx, cy))
                motion_level, motion_threat, motion_color = motion_tracker.get_motion_level()

                # Toplam tehdit
                total_threat = min(100, base_threat + motion_threat)

                # Niyet
                intent = get_intent(gesture, total_threat)

                # Kutu
                x1 = int(min(x_list)*w) - 20
                x2 = int(max(x_list)*w) + 20
                y1 = int(min(y_list)*h) - 20
                y2 = int(max(y_list)*h) + 20

                # Tehdit seviyesine göre kutu rengi
                if total_threat >= 70:
                    box_color = (0, 0, 255)
                elif total_threat >= 40:
                    box_color = (0, 165, 255)
                else:
                    box_color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(frame, f"{gesture}", (x1, y1-35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame, f"{motion_level} ({int(speed)})", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)

                # Tehdit kaydı
                if total_threat >= 70:
                    if not recorder.is_recording():
                        recorder.start_recording()

                    if time.time() - last_alert > 3:
                        events.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "gesture": gesture,
                            "threat": total_threat
                        })
                        last_alert = time.time()

        # Video kaydı güncelle
        is_recording = recorder.update(frame)

        # Kayıt göstergesi
        if is_recording:
            if frame_count % 10 < 5:
                cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (50, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # === PANEL ===
        panel_h = 180
        panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        # Başlık
        cv2.putText(panel, "MEDIAPIPE EL ANALIZI - GELISMIS", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(panel, (10, 35), (w-10, 35), (60, 60, 60), 1)

        # Sol: Hareket bilgisi
        cv2.putText(panel, f"Hareket: {gesture}", (15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(panel, f"Parmak: {fingers}", (15, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(panel, f"Hiz: {motion_level}", (15, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
        cv2.putText(panel, f"({int(speed)} px/s)", (15, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Orta: Niyet kutusu
        intent_color = (0, 0, 255) if total_threat >= 70 else \
                      (0, 165, 255) if total_threat >= 40 else (0, 255, 0)
        cv2.rectangle(panel, (200, 50), (420, 95), intent_color, -1)
        cv2.putText(panel, intent, (210, 82),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Tehdit çubuğu
        cv2.putText(panel, "Tehdit:", (200, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        bar_x, bar_y, bar_w, bar_h = 260, 105, 160, 18
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * total_threat / 100)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), intent_color, -1)
        cv2.putText(panel, f"{total_threat}%", (bar_x+bar_w+10, bar_y+14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, intent_color, 2)

        # Sağ: Durum
        status_x = w - 120
        if total_threat >= 70:
            if frame_count % 8 < 4:
                cv2.circle(panel, (status_x, 70), 20, (0, 0, 255), -1)
            cv2.putText(panel, "ALARM!", (status_x-35, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif total_threat >= 40:
            cv2.circle(panel, (status_x, 70), 20, (0, 165, 255), -1)
            cv2.putText(panel, "DIKKAT", (status_x-35, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            cv2.circle(panel, (status_x, 70), 20, (0, 255, 0), -1)
            cv2.putText(panel, "GUVENLI", (status_x-40, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Alt: Olaylar
        cv2.putText(panel, "Son Tehditler:", (15, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        for i, ev in enumerate(list(events)[-3:]):
            txt = f"[{ev['time']}] {ev['gesture']} %{ev['threat']}"
            cv2.putText(panel, txt, (130 + i*160, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1)

        # Kayıt bilgisi
        if is_recording:
            cv2.putText(panel, "KAYIT YAPILIYOR...", (w-180, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Birleştir
        combined = np.vstack([frame, panel])
        cv2.imshow('MediaPipe El Analizi - Gelismis', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Temizlik
    if recorder.is_recording():
        recorder.stop_recording()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print("\n" + "=" * 55)
    print("OTURUM RAPORU")
    print("=" * 55)
    print(f"Kayıt klasörü: {RECORD_DIR}/")
    print(f"Toplam tehdit olayı: {len(events)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
