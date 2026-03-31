require("dotenv").config();

const express = require("express");
const cors = require("cors");
const { createClient } = require("@supabase/supabase-js");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY,
  {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  }
);

app.get("/", (req, res) => {
  res.json({
    ok: true,
    message: "Sensor backend is running",
  });
});

app.post("/api/readings", async (req, res) => {
  try {
    console.log("BODY RECEIVED:", req.body);

    const { device_id, temperature, humidity, airQuality } = req.body;

    if (
      temperature === undefined ||
      humidity === undefined ||
      airQuality === undefined
    ) {
      return res.status(400).json({
        ok: false,
        error: "Missing temperature, humidity, or airQuality",
      });
    }

    const payload = {
      device_id: device_id || "arduino-uno-01",
      temperature_c: Number(temperature),
      humidity: Number(humidity),
      air_quality: Number(airQuality),
    };

    console.log("INSERTING:", payload);

    const { data, error } = await supabase
      .from("sensor_readings")
      .insert([payload])
      .select();

    if (error) {
      console.error("Supabase insert error:", error);
      return res.status(500).json({
        ok: false,
        error: error.message,
      });
    }

    console.log("INSERT SUCCESS:", data[0]);

    return res.status(200).json({
      ok: true,
      inserted: data[0],
    });
  } catch (err) {
    console.error("POST /api/readings error:", err);
    return res.status(500).json({
      ok: false,
      error: "Internal server error",
    });
  }
});

app.get("/api/readings", async (req, res) => {
  try {
    const limit = Number(req.query.limit || 100);

    const { data, error } = await supabase
      .from("sensor_readings")
      .select("*")
      .order("recorded_at", { ascending: false })
      .limit(limit);

    if (error) {
      console.error("Supabase read error:", error);
      return res.status(500).json({
        ok: false,
        error: error.message,
      });
    }

    return res.json({
      ok: true,
      count: data.length,
      readings: data,
    });
  } catch (err) {
    console.error("GET /api/readings error:", err);
    return res.status(500).json({
      ok: false,
      error: "Internal server error",
    });
  }
});

app.get("/api/latest", async (req, res) => {
  try {
    const { data, error } = await supabase
      .from("sensor_readings")
      .select("*")
      .order("recorded_at", { ascending: false })
      .limit(1);

    if (error) {
      console.error("Supabase latest error:", error);
      return res.status(500).json({
        ok: false,
        error: error.message,
      });
    }

    return res.json({
      ok: true,
      latest: data[0] || null,
    });
  } catch (err) {
    console.error("GET /api/latest error:", err);
    return res.status(500).json({
      ok: false,
      error: "Internal server error",
    });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Open: http://localhost:${PORT}/`);
});