using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging;
using System.Net;

namespace PlaywrightMultiTest
{
    public class StaticFileServer
    {
        WebApplication? app;
        Task? runningTask;
        string WWWRoot;
        string RequestPath;
        string Url;
        string devcertPath;
        public StaticFileServer(string wwwroot, string url, string requestPath = "")
        {
            if (string.IsNullOrEmpty(wwwroot))
            {
                throw new ArgumentNullException(nameof(wwwroot));
            }
            if (!Directory.Exists(wwwroot))
            {
                throw new DirectoryNotFoundException(wwwroot);
            }
            WWWRoot = Path.GetFullPath(wwwroot);
            RequestPath = requestPath;
            Url = url;
            devcertPath = Path.GetFullPath("assets/testcert.pfx");
            if (!File.Exists(devcertPath))
                throw new Exception("testcert.pfx not found. Cannot create static server");
        }
        public bool Running => runningTask?.IsCompleted == false;
        public void Start()
        {
            runningTask ??= StartAsync();
        }
        private async Task StartAsync()
        {
            try
            {
                var builder = WebApplication.CreateBuilder();
                var port = new Uri(Url).Port;

                // This wipes out Console, Debug, and any other default providers
                builder.Logging.ClearProviders();

                // Configure static file serving
                builder.WebHost.UseKestrel();
                builder.WebHost.ConfigureKestrel(serverOptions =>
                {
                    serverOptions.Listen(IPAddress.Loopback, port, listenOptions =>
                    {
                        listenOptions.UseHttps(devcertPath, "unittests");
                    });
                });
                // Use the current directory as the web root
                builder.Environment.WebRootPath = WWWRoot;
                builder.WebHost.UseUrls(Url);

                app = builder.Build();

                // (optional) add headers that enables: window.crossOriginIsolated == true
                app.Use(async (context, next) =>
                {
                    context.Response.Headers["Cross-Origin-Embedder-Policy"] = "credentialless";
                    context.Response.Headers["Cross-Origin-Opener-Policy"] = "same-origin";
                    await next();
                });

                // Test OUTPUT sink: POST /__pmt/out/<relative path> writes the body to
                // _mldump/test-out/<relative path>. A browser test has no file system, and
                // Console.WriteLine output is summarised away, so without this a WebGPU test can
                // compare against a reference but can never hand its raw tensors back for a
                // side-by-side (the DAv3 ORT/Transformers.js parity images are the first user).
                // Loopback only (see Listen above); the path is confined to test-out/.
                app.MapPost("/__pmt/out/{**name}", async (Microsoft.AspNetCore.Http.HttpContext ctx, string name) =>
                {
                    var root = Path.GetFullPath(Path.Combine(TestResultsWriter.MlDumpDir, "test-out"));
                    var dest = Path.GetFullPath(Path.Combine(root, name));
                    if (!dest.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                        return Microsoft.AspNetCore.Http.Results.BadRequest("path escapes test-out");
                    Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
                    await using (var fs = File.Create(dest))
                        await ctx.Request.Body.CopyToAsync(fs);
                    return Microsoft.AspNetCore.Http.Results.Ok(dest);
                });

                // enable 404 fallback to default root
                app.UseStatusCodePagesWithReExecute(string.IsNullOrEmpty(RequestPath) ? "/" : RequestPath);

                // enable index.html fallback
                app.UseDefaultFiles(new DefaultFilesOptions
                {
                    FileProvider = new PhysicalFileProvider(WWWRoot),
                    RequestPath = RequestPath
                });
                // enable unknown file types (required)
                app.UseFileServer(new FileServerOptions
                {
                    FileProvider = new PhysicalFileProvider(WWWRoot),
                    RequestPath = RequestPath,
                    EnableDirectoryBrowsing = false, // Optional: allows browsing directory listings
                    StaticFileOptions = {
                        ServeUnknownFileTypes = true, // Crucial: serves all file types, even those without known MIME types
                        DefaultContentType = "application/octet-stream" // Optional: default MIME type for unknown files
                    }
                });
                // start hosting
                await app.RunAsync();
            }
            finally
            {
                app = null;
                runningTask = null;
            }
        }
        public async Task Stop()
        {
            if (app == null || runningTask == null) return;
            try
            {
                await app.StopAsync();
            }
            catch { }
            await app.DisposeAsync();
            if (runningTask != null)
            {
                try
                {
                    await runningTask;
                }
                catch { }
            }
        }
    }
}
