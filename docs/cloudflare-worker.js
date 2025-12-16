/**
 * TinyGPU Gemini API Proxy - Cloudflare Worker
 *
 * This worker proxies requests to the Gemini API, keeping your API key secure.
 * Deploy this to Cloudflare Workers and set the GEMINI_API_KEY secret.
 *
 * Setup Instructions:
 * 1. Go to https://dash.cloudflare.com/ and sign up/login
 * 2. Go to Workers & Pages > Create Application > Create Worker
 * 3. Name it something like "tinygpu-gemini-proxy"
 * 4. Replace the default code with this file's contents
 * 5. Go to Settings > Variables > Add Variable
 *    - Name: GEMINI_API_KEY
 *    - Value: Your Gemini API key
 *    - Click "Encrypt" to keep it secret
 * 6. Save and Deploy
 * 7. Your worker URL will be: https://tinygpu-gemini-proxy.<your-subdomain>.workers.dev
 */

export default {
  async fetch(request, env) {
    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
          "Access-Control-Max-Age": "86400",
        },
      });
    }

    // Only allow POST requests
    if (request.method !== "POST") {
      return new Response(JSON.stringify({ error: "Method not allowed" }), {
        status: 405,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }

    try {
      // Get the request body
      const body = await request.json();

      // Validate required fields
      if (!body.prompt) {
        return new Response(JSON.stringify({ error: "Missing prompt" }), {
          status: 400,
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
          },
        });
      }

      // Build Gemini API request
      const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${env.GEMINI_API_KEY}`;

      const geminiPayload = {
        contents: [{ parts: [{ text: body.prompt }] }],
      };

      // Add system instruction if provided
      if (body.systemPrompt) {
        geminiPayload.systemInstruction = {
          parts: [{ text: body.systemPrompt }],
        };
      }

      // Call Gemini API
      const geminiResponse = await fetch(geminiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(geminiPayload),
      });

      if (!geminiResponse.ok) {
        const errorText = await geminiResponse.text();
        return new Response(
          JSON.stringify({
            error: "Gemini API error",
            status: geminiResponse.status,
            details: errorText,
          }),
          {
            status: geminiResponse.status,
            headers: {
              "Content-Type": "application/json",
              "Access-Control-Allow-Origin": "*",
            },
          }
        );
      }

      const data = await geminiResponse.json();
      const text =
        data.candidates?.[0]?.content?.parts?.[0]?.text ||
        "No response generated.";

      return new Response(JSON.stringify({ text }), {
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    } catch (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }
  },
};
