namespace Phi3Onnx
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            var path = @"<root>\Phi-3-mini-4k-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32";

            var bot = new BotService(path);

            var response = await bot.QueryAsync("What is a ferrari ?");

            Console.Write(response);

            Console.ReadLine();
        }
    }
}
