using Microsoft.Playwright;

namespace PlaywrightMultiTest
{
    public static class IPageExtensions
    {
        public static async Task WaitForConditionAsync(this IPage page, Func<Task<bool>> condition, int timeoutMs = 120_000)
        {
            var start = DateTime.Now;
            while ((DateTime.Now - start).TotalMilliseconds < timeoutMs)
            {
                if (await condition()) return;
                await Task.Delay(100); // Polling interval
            }
            throw new TimeoutException("Condition was not met within the timeout.");
        }
    }
}
