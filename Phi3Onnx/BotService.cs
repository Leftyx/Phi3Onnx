using System.Diagnostics;
using System.Text;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace Phi3Onnx
{
    internal class BotService
    {
        private readonly Model _model;
        private readonly Tokenizer _tokenizer;
        private const string SYSTEM_PROMPT = "You are a knowledgeable assistant and you will answer the following question as clearly and concisely as possible, providing only the relevant information requested.";

        public BotService(string modelPath)
        {
            _model = new Model(modelPath);
            _tokenizer = new Tokenizer(_model);
        }

        public async Task<string> QueryAsync(string userPrompt)
        {
            if (_model is null)
            {
                return string.Empty;
            }

            StringBuilder builder = new();

            var prompt = $@"<|system|>{SYSTEM_PROMPT}<|end|><|user|>{userPrompt}<|end|><|assistant|>";

            await foreach (var part in InferStreaming(prompt))
            {
                builder.Append(part);
            }

            return builder.ToString();
        }

        public async IAsyncEnumerable<string> InferStreaming(string prompt)
        {
            if (_model == null || _tokenizer == null)
            {
                throw new InvalidOperationException("Model is not ready");
            }

            var generatorParams = new GeneratorParams(_model);

            var sequences = _tokenizer.Encode(prompt);

            generatorParams.SetSearchOption("max_length", 4096);
            generatorParams.SetInputSequences(sequences);
            generatorParams.TryGraphCaptureWithMaxBatchSize(1);

            using var tokenizerStream = _tokenizer.CreateStream();
            using var generator = new Generator(_model, generatorParams);

            StringBuilder builder = new();

            while (!generator.IsDone())
            {
                string part;

                try
                {
                    await Task.Delay(10).ConfigureAwait(false);

                    generator.ComputeLogits();
                    generator.GenerateNextToken();

                    part = tokenizerStream.Decode(generator.GetSequence(0)[^1]);

                    builder.Append(part);

                    var builderAsString = builder.ToString();

                    if (builderAsString.Contains("<|end|>")
                        || builderAsString.Contains("<|user|>")
                        || builderAsString.Contains("<|system|>"))
                    {
                        yield break;
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex);
                    yield break;
                }

                yield return part;
            }
        }
    }
}
