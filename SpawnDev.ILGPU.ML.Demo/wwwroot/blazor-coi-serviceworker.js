/*! coi-serviceworker - Cross-Origin Isolation via Service Worker */
/*
 * This service worker intercepts all responses and adds the required
 * Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers
 * to enable SharedArrayBuffer support in the browser.
 *
 * Works locally and on static hosts like GitHub Pages where you
 * cannot set server response headers.
 *
 * Registration: Add <script src="blazor-coi-serviceworker.js"></script> to index.html
 * The static <script src="_framework/blazor.webassembly.js"> tag MUST also be in index.html
 * as the fallback — Blazor always loads, COI just adds SharedArrayBuffer support.
 */

if (typeof window !== 'undefined') {
    // --- Running as a regular script in the page context ---

    var verbose = false;
    function consoleLog(...args) {
        if (!verbose) return;
        console.log("[COI]", ...args);
    }

    if (window.crossOriginIsolated) {
        // Already cross-origin isolated — SharedArrayBuffer available
        consoleLog("[COI] Cross-origin isolated ✓");
    } else if ("serviceWorker" in navigator) {
        // Not yet isolated — register/activate the SW, then reload ONCE to apply headers.
        // Use sessionStorage to prevent infinite reload loops: if COI still fails after
        // one reload (browser quirk, credentialless unsupported, etc.), stop retrying
        // and let Blazor load without SharedArrayBuffer rather than hanging.
        var reloadKey = "coi-reload-count";
        var reloadCount = parseInt(sessionStorage.getItem(reloadKey) || "0", 10);

        if (reloadCount < 2) {
            // Register the SW (idempotent if already registered)
            navigator.serviceWorker
                .register(window.document.currentScript.src)
                .then(function (reg) {
                    consoleLog("[COI] Service worker registered:", reg.scope);
                })
                .catch(function (err) {
                    console.error("[COI] Service worker registration failed:", err);
                });

            // Wait for SW to be ready, then reload to pick up COI headers.
            // Timeout after 5s — if the SW doesn't activate in time, let Blazor load anyway.
            var reloaded = false;
            var doReload = function () {
                if (reloaded) return;
                reloaded = true;
                sessionStorage.setItem(reloadKey, String(reloadCount + 1));
                consoleLog("[COI] Reloading to apply COI headers (attempt " + (reloadCount + 1) + ")");
                window.location.reload();
            };

            navigator.serviceWorker.ready.then(doReload);
            setTimeout(function () {
                if (!reloaded && navigator.serviceWorker.controller) {
                    // SW is controlling but ready didn't fire — force reload
                    doReload();
                } else if (!reloaded) {
                    consoleLog("[COI] Service worker not ready after 5s — loading without COI");
                }
            }, 5000);
        } else {
            // Already tried reloading twice — COI isn't working, proceed without it.
            // Clear the counter so next fresh navigation can try again.
            console.warn("[COI] Cross-origin isolation failed after " + reloadCount +
                " reload(s) — SharedArrayBuffer unavailable. Wasm limited to 1 worker.");
            sessionStorage.removeItem(reloadKey);
        }
    } else {
        consoleLog("[COI] Service workers not supported — SharedArrayBuffer unavailable");
    }

    // On successful COI, clear the reload counter so future refreshes work cleanly.
    if (window.crossOriginIsolated) {
        sessionStorage.removeItem("coi-reload-count");
    }
} else {
    // --- Running as a Service Worker ---
    var verbose = false;
    function consoleLog(...args) {
        if (!verbose) return;
        console.log("[COI]", ...args);
    }

    self.addEventListener("install", function () { self.skipWaiting(); });
    self.addEventListener("activate", function (event) { event.waitUntil(self.clients.claim()); });

    self.addEventListener("fetch", function (event) {
        // Skip requests the SW can't meaningfully intercept
        if (event.request.cache === "only-if-cached" && event.request.mode !== "same-origin") {
            return;
        }

        // Don't intercept cross-origin requests — we can't add useful
        // headers to them and they are the main source of "Failed to fetch" errors
        var url = new URL(event.request.url);
        if (url.origin !== self.location.origin) {
            return; // Let the browser handle it natively
        }

        event.respondWith(
            fetch(event.request)
                .then(function (response) {
                    var newHeaders = new Headers(response.headers);
                    newHeaders.set("Cross-Origin-Embedder-Policy", "credentialless");
                    newHeaders.set("Cross-Origin-Opener-Policy", "same-origin");

                    return new Response(response.body, {
                        status: response.status,
                        statusText: response.statusText,
                        headers: newHeaders,
                    });
                })
                .catch(function (e) {
                    consoleLog("[COI] Fetch failed for:", event.request.url, e.message);
                    return new Response("Service Worker fetch failed", {
                        status: 502,
                        statusText: "Service Worker Fetch Failed",
                    });
                })
        );
    });
}
