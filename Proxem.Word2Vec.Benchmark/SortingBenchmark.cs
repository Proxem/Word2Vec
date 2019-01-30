using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using Proxem.BlasNet;
using Proxem.NumNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Proxem.Word2Vec.Benchmark
{
    [CoreJob]
    [GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
    [CategoriesColumn]
    public class SortBenchmarkMkl
    {
        Word2Vec _w2v;
        private Array<float> vector_test;
        private Array<float> matrix_test;

        private float[][] bestd_v;
        private int[][] bestw_v;
        private float[][] bestd_m;
        private int[][] bestw_m;

        [Params(100, 500)]
        public int N;

        [Params(100000, 2000000)]
        public int Vocab;

        [Params(10, 100)]
        public int Neighbors;

        [GlobalSetup]
        public void Setup()
        {
            // Launching mkl for NumNet (path might need to be change)
            var path = "C:/data/dlls/mkl";
            StartProvider.LaunchMklRt(1, path);

            var m = NN.Random.Normal(0f, 0.1f, Vocab, N);
            var vocab = (from i in Enumerable.Range(0, Vocab)
                         select i.ToString()).ToArray();
            _w2v = new Word2Vec(vocab, m);

            int batch = 50;

            vector_test = NN.Random.Normal(0, 0.1f, N, 1);
            matrix_test = NN.Random.Normal(0, 0.1f, N, batch);

            bestd_v = new float[1][];
            bestw_v = new int[1][];
            bestd_v[0] = new float[Neighbors];
            bestw_v[0] = new int[Neighbors];
            bestd_m = new float[batch][];
            bestw_m = new int[batch][];
            for (int i = 0; i < batch; i++)
            {
                bestd_m[i] = new float[Neighbors];
                bestw_m[i] = new int[Neighbors];
            }
        }

        [BenchmarkCategory("Sort Vector"), Benchmark(Baseline = true)]
        public void BaseSortingV()
        {
            _w2v.NBest(vector_test, bestd_v[0], bestw_v[0]);
        }

        [BenchmarkCategory("Sort Vector")]
        public void ClassicalSortingV()
        {
            _w2v.NBestClassic(vector_test, bestd_v[0], bestw_v[0]);
        }

        [BenchmarkCategory("Sort Vector")]
        public void QuicksortSortingV()
        {
            _w2v.NBestQs(vector_test, bestd_v[0], bestw_v[0]);
        }

        [BenchmarkCategory("Sort Vector")]
        public void HeapsortSortingV()
        {
            _w2v.NBestHeap(vector_test, bestd_v[0], bestw_v[0]);
        }

        [BenchmarkCategory("Sort Vector")]
        public void HeapsortSortingBatchV()
        {
            _w2v.NBestHeap(vector_test, bestd_v, bestw_v);
        }

        [BenchmarkCategory("Sort Vector")]
        public void HeapsortSortingBatchParallelV()
        {
            _w2v.NBestHeapParallel(vector_test, bestd_v, bestw_v);
        }

        [BenchmarkCategory("Sort Batch"), Benchmark(Baseline = true)]
        public void BaseSortingM()
        {
            for (int i = 0; i < bestd_m.Length; i++)
            {
                _w2v.NBest(matrix_test[Slicer._, i], bestd_m[i], bestw_m[i]);
            }
        }

        [BenchmarkCategory("Sort Batch")]
        public void ClassicalSortingM()
        {
            for (int i = 0; i < bestd_m.Length; i++)
            {
                _w2v.NBest(matrix_test[Slicer._, i], bestd_m[i], bestw_m[i]);
            }
        }

        //[BenchmarkCategory("Sort Batch")]
        //public void ClassicalSortingBatchM()
        //{
        //    // TODO : implement batch classical sorting method
        //}

        [BenchmarkCategory("Sort Batch")]
        public void QuicksortSortingM()
        {
            for (int i = 0; i < bestd_m.Length; i++)
            {
                _w2v.NBestQs(matrix_test[Slicer._, i], bestd_m[i], bestw_m[i]);
            }
        }

        //[BenchmarkCategory("Sort Batch")]
        //public void QuicksortSortingbatchM()
        //{
        //    // TODO : implement batch sorting for quicksort
        //}

        [BenchmarkCategory("Sort Batch")]
        public void HeapsortSortingM()
        {
            for (int i = 0; i < bestd_m.Length; i++)
            {
                _w2v.NBestHeap(matrix_test[Slicer._, i], bestd_m[i], bestw_m[i]);
            }
        }

        [BenchmarkCategory("Sort Batch")]
        public void HeapsortSortingBatchM()
        {
            _w2v.NBestHeap(matrix_test, bestd_m, bestw_m);
        }

        [BenchmarkCategory("Sort Batch")]
        public void HeapsortSortingBatchParallelM()
        {
            _w2v.NBestHeapParallel(matrix_test, bestd_m, bestw_m);
        }
    }
}
