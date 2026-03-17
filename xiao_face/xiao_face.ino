#include <Arduino_GFX_Library.h>
// Camera support will be added in a future update

// --- Verified Landmark Mapping (USB UP) ---
// TOP LEFT: VUSB | TOP RIGHT: 5V/VCC
// GND is directly below VUSB on the LEFT rail.

#define BLACK 0x0000

/* --- LCD Pins --- */
#define SCK_PIN 8   // D8
#define MOSI_PIN 9  // D9
#define CS_PIN 3    // D3
#define DC_PIN 2    // D2
#define RST_PIN 1   // D1

/* --- Serial Synapse --- */
// Pi Pin 8 (TX) -> XIAO D7 (RX)
// Pi Pin 10 (RX) -> XIAO D6 (TX)99999

Arduino_DataBus *bus = new Arduino_ESP32SPI(DC_PIN, CS_PIN, SCK_PIN, MOSI_PIN);
Arduino_GFX *gfx = new Arduino_GC9A01(bus, RST_PIN, 0 /* rotation */, true /* IPS */);

float currentZeta = 0.88;
float currentKappa = 0.99;
float pulseWave = 0;

void setup() {
  Serial.begin(115200);   // USB Debug
  Serial1.begin(115200, SERIAL_8N1, 7, 6); // D7 is RX, D6 is TX (The Synapse)

  gfx->begin();
  gfx->fillScreen(BLACK);
  
  // Initialize Camera (The Eye)
  // We will expand this for the vision stream next
  Serial.println("[HFF] Face Initialized.");
}

void loop() {
  // Listen for the Pi's Pulse
  if (Serial1.available()) {
    String packet = Serial1.readStringUntil('\n');
    if (packet.startsWith("Z:")) {
      int pipe = packet.indexOf('|');
      currentZeta = packet.substring(2, pipe).toFloat();
      currentKappa = packet.substring(pipe + 3).toFloat();
    }
  }

  // Draw the Radial Resonance
  int centerX = 120;
  int centerY = 120;
  
  // Erase old pulse
  gfx->drawCircle(centerX, centerY, (int)pulseWave, BLACK);

  // Calculate new wave: Zeta controls frequency, Kappa controls amplitude
  pulseWave = 60 + (sin(millis() * 0.005 * currentZeta) * 30 * currentKappa);

  // Render Crystalline Blue Pulse
  uint16_t color = gfx->color565(0, 150 + (currentZeta * 100), 255);
  gfx->drawCircle(centerX, centerY, (int)pulseWave, color);
  gfx->drawCircle(centerX, centerY, (int)pulseWave - 1, color);

  delay(5); // Survival Latency
}
