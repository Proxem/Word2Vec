using BenchmarkDotNet.Running;
using System;

namespace Proxem.Word2Vec.Benchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            var summary = BenchmarkRunner.Run<SortBenchmarkMkl>();
            Console.ReadLine();
        }
    }
}
