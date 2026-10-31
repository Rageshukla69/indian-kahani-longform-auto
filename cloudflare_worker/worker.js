/**
 * Free AI Image Generation API using Cloudflare Workers AI (v2.0 — Photorealistic 16:9 Edition)
 * Based on https://github.com/saurav-z/free-image-generation-api
 *
 * Supports:
 * - @cf/black-forest-labs/flux-1-schnell (Photorealistic, ~173 free images/day on 10,000 free daily neurons)
 * - @cf/bytedance/stable-diffusion-xl-lightning (Native 1344x768 16:9 widescreen generation)
 * - @cf/stabilityai/stable-diffusion-xl-base-1.0 (Native 1344x768 16:9 widescreen generation)
 */

export default {
  async fetch(request, env) {
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    if (request.method === "GET") {
      return new Response(
        JSON.stringify({
          status: "ok",
          service: "free-image-generation-api (Indian Kahani Photorealistic 16:9 Edition)",
          repo: "https://github.com/saurav-z/free-image-generation-api",
          models: [
            "@cf/black-forest-labs/flux-1-schnell",
            "@cf/bytedance/stable-diffusion-xl-lightning",
            "@cf/stabilityai/stable-diffusion-xl-base-1.0",
          ],
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    if (request.method !== "POST") {
      return new Response(JSON.stringify({ error: "Method Not Allowed. Use POST." }), {
        status: 405,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const expectedKey = (env.API_KEY || "").trim();
    if (expectedKey) {
      const authHeader = request.headers.get("Authorization") || "";
      const xApiKey = request.headers.get("x-api-key") || "";
      const token = authHeader.replace(/^Bearer\s+/i, "").trim() || xApiKey.trim();
      if (token !== expectedKey) {
        return new Response(JSON.stringify({ error: "Unauthorized: Invalid API_KEY" }), {
          status: 401,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
    }

    if (!env.AI) {
      return new Response(
        JSON.stringify({
          error: "Workers AI binding 'AI' is missing! Go to Worker Settings -> Bindings -> Add Workers AI binding named 'AI'.",
        }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    try {
      const body = await request.json();
      const rawPrompt = (body.prompt || "").trim();
      if (!rawPrompt) {
        return new Response(JSON.stringify({ error: "Missing 'prompt' in JSON body" }), {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      // Enforce photorealistic 16:9 widescreen composition in prompt
      const prompt = rawPrompt.includes("photorealistic")
        ? rawPrompt
        : `${rawPrompt}, ultra-realistic 8k cinematic wildlife and historical live-action photography, shot on ARRI Alexa 65, 35mm anamorphic lens, natural photorealistic textures, wide 16:9 horizontal widescreen framing, NOT cartoon, NOT 2D illustration, no text, no watermarks`;

      const model = body.model || "@cf/black-forest-labs/flux-1-schnell";
      const width = Number(body.width) || 1344;
      const height = Number(body.height) || 768;
      const steps = Number(body.num_steps || body.steps) || 4;

      let inputs = { prompt };
      if (model.includes("flux-1-schnell")) {
        inputs = { prompt, steps: Math.min(8, Math.max(4, steps)) };
      } else {
        inputs = {
          prompt,
          negative_prompt: "cartoon, anime, 2d illustration, comic, vector, drawing, sketch, deformed, blurry, watermark, text",
          width,
          height,
          num_steps: Math.min(20, Math.max(4, steps)),
        };
      }

      const response = await env.AI.run(model, inputs);

      if (response && typeof response === "object" && "image" in response && typeof response.image === "string") {
        const binaryString = atob(response.image);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        return new Response(bytes.buffer, {
          headers: { ...corsHeaders, "Content-Type": "image/jpeg" },
        });
      }

      return new Response(response, {
        headers: { ...corsHeaders, "Content-Type": "image/png" },
      });
    } catch (err) {
      return new Response(JSON.stringify({ error: String(err) }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
  },
};
