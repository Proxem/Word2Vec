using Proxem.NumNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;

namespace Proxem.Word2Vec.Test
{
    public class TestSortingNeighbors: IClassFixture<Shared>
    {
        private Word2Vec _w2v; // random matrix of shape (10, 4)

        private readonly Array<float> _test_1; // random vector of shape (1, 4)
        private readonly Array<float> _test_2; // random vector of shape (1, 4)
        private Array<float> _test_3; // [_test_1, _test_2] concatenation, shape (4, 2)

        public TestSortingNeighbors(Shared shared)
        {
            this._w2v = shared.W2v;
            this._test_1 = shared.Test1;
            this._test_2 = shared.Test2;
            this._test_3 = shared.Test3;
        }

        [Fact]
        public void TestBaseline()
        {
            var bestd = new float[4];
            var bestw = new int[4];

            var expectedd = new float[4] { 1.34889853f, 1.17417169f, 1.15919185f, 0.866888f };
            var expectedw = new int[4] { 8, 2, 0, 5 };

            _w2v.NBest(_test_1, bestd, bestw);
            AssertArray.AreAlmostEqual(expectedd, bestd);
            AssertArray.AreEqual(expectedw, bestw);
        }

        [Fact]
        public void TestBuildVPTree()
        {
            _w2v.NBestVPTree(_test_1, new double[5], new int[5]);
        }

        [Fact]
        public void TestVPTree()
        {
            var bestd = new double[4];
            var bestw = new int[4];

            // with this test the order is not exactly the same due to the vector normalisation required by the VPTree
            // this is not a problem as it is often better to normalize before doing the neighbors search
            var expectedd = new double[4] { 0.132890641689301, 0.480090379714966, 0.648832619190216, 0.662571460008621 };
            var expectedw = new int[4] { 8, 2, 5, 0 };

            _w2v.NBestVPTree(_test_1, bestd, bestw);
            AssertArray.AreAlmostEqual(expectedd, bestd);
            AssertArray.AreEqual(expectedw, bestw);
        }

        [Fact]
        public void TestHeapsortSingle()
        {
            var bestd = new float[4];
            var bestw = new int[4];

            var expectedd = new float[4] { 1.34889853f, 1.17417169f, 1.15919185f, 0.866888f };
            var expectedw = new int[4] { 8, 2, 0, 5 };
            _w2v.NBestHeap(_test_1, bestd, bestw);
            AssertArray.AreAlmostEqual(expectedd, bestd);
            AssertArray.AreEqual(expectedw, bestw);

            expectedd = new float[4] { 4.63800335f, 4.19298649f, 3.162262f, 1.9904952f };
            expectedw = new int[4] { 1, 7, 3, 4 };
            _w2v.NBestHeap(_test_2, bestd, bestw);
            AssertArray.AreAlmostEqual(expectedd, bestd);
            AssertArray.AreEqual(expectedw, bestw);
        }

        [Fact]
        public void TestHeapsortBatch()
        {
            var bestd = new float[_test_3.Shape[1]][];
            var bestw = new int[_test_3.Shape[1]][];
            int neighbors = 3;
            for (int i = 0; i < bestd.Length; i++)
            {
                bestd[i] = new float[neighbors];
                bestw[i] = new int[neighbors];
            }

            var expectedd = new float[6] { 1.34889853f, 1.17417169f, 1.15919185f, 4.63800335f, 4.19298649f, 3.162262f };
            var expectedw = new int[6] { 8, 2, 0, 1, 7, 3 };
            _w2v.NBestHeap(_test_3, bestd, bestw);
            AssertArray.AreAlmostEqual(expectedd, (from bd in bestd
                                                   from d in bd
                                                   select d).ToArray());
            AssertArray.AreEqual(expectedw, (from bw in bestw
                                             from w in bw
                                             select w).ToArray());
        }

        [Fact]
        public void TestHeapsortBatchParallel()
        {
            var bestd = new float[_test_3.Shape[1]][];
            var bestw = new int[_test_3.Shape[1]][];
            int neighbors = 3;
            for (int i = 0; i < bestd.Length; i++)
            {
                bestd[i] = new float[neighbors];
                bestw[i] = new int[neighbors];
            }

            var expectedd = new float[6] { 1.34889853f, 1.17417169f, 1.15919185f, 4.63800335f, 4.19298649f, 3.162262f };
            var expectedw = new int[6] { 8, 2, 0, 1, 7, 3 };
            _w2v.NBestHeapParallel(_test_3, bestd, bestw);
            AssertArray.AreAlmostEqual(expectedd, (from bd in bestd
                                                   from d in bd
                                                   select d).ToArray());
            AssertArray.AreEqual(expectedw, (from bw in bestw
                                             from w in bw
                                             select w).ToArray());
        }

        [Fact]
        public void TestCompatibilityHeapsortBatchSingle()
        {
            // test that batch and single line heapsort have same results

            var test = NN.Random.Normal(0.1f, 0.1f, 4, 20);
            var bestd = new float[test.Shape[1]][];
            var bestw = new int[test.Shape[1]][];
            int neighbors = 3;
            for (int i = 0; i < bestd.Length; i++)
            {
                bestd[i] = new float[neighbors];
                bestw[i] = new int[neighbors];
            }

            _w2v.NBestHeap(test, bestd, bestw);

            var singleBestd = new float[neighbors];
            var singleBestw = new int[neighbors];
            for (int i = 0; i < test.Shape[1]; i++)
            {
                _w2v.NBestHeap(test[Slicer._, i], singleBestd, singleBestw);
                AssertArray.AreEqual(singleBestw, bestw[i]);
                AssertArray.AreAlmostEqual(singleBestd, bestd[i]);
            }
        }

        [Fact]
        public void TestCompatibilityHeapsortBatchBatchParallel()
        {
            // test that heapsort batch parallel and batch have same results
            var test = NN.Random.Normal(0f, 0.1f, 4, 30);
            
            var bestd_1 = new float[test.Shape[1]][];
            var bestd_2 = new float[test.Shape[1]][];
            var bestw_1 = new int[test.Shape[1]][];
            var bestw_2 = new int[test.Shape[1]][];
            int neighbors = 6;
            for (int i = 0; i < test.Shape[1]; i++)
            {
                bestd_1[i] = new float[neighbors];
                bestd_2[i] = new float[neighbors];
                bestw_1[i] = new int[neighbors];
                bestw_2[i] = new int[neighbors];
            }

            _w2v.NBestHeap(test, bestd_1, bestw_1);
            _w2v.NBestHeapParallel(test, bestd_2, bestw_2);

            for (int i = 0; i < test.Shape[1]; i++)
            {
                AssertArray.AreAlmostEqual(bestd_1[i], bestd_2[i]);
                AssertArray.AreEqual(bestw_1[i], bestw_2[i]);
            }
        }

        [Fact]
        public void TestCompatibilityHeapsortSort()
        {
            // test that heapsort single and sort single have same results

            var test = NN.Random.Normal(0f, 0.1f, 4, 10);
            int neighbors = 6;
            var bestd_1 = new float[neighbors];
            var bestd_2 = new float[neighbors];
            var bestw_1 = new int[neighbors];
            var bestw_2 = new int[neighbors];
            for (int i = 0; i < test.Shape[1]; i++)
            {
                _w2v.NBestHeap(test[Slicer._, i], bestd_1, bestw_1);
                _w2v.NBest(test[Slicer._, i], bestd_2, bestw_2);
                AssertArray.AreAlmostEqual(bestd_1, bestd_2);
                AssertArray.AreEqual(bestw_1, bestw_2);
            }
        }

        [Fact]
        public void TestQuicksortSingle()
        {
            var bestd = new float[4];
            var bestw = new int[4];

            var expectedd = new float[4] { 1.34889853f, 1.17417169f, 1.15919185f, 0.866888f };
            var expectedw = new int[4] { 8, 2, 0, 5 };
            _w2v.NBestQs(_test_1, bestd, bestw);
            AssertArray.AreAlmostEqual(expectedd, bestd);
            AssertArray.AreEqual(expectedw, bestw);

            expectedd = new float[4] { 4.63800335f, 4.19298649f, 3.162262f, 1.9904952f };
            expectedw = new int[4] { 1, 7, 3, 4 };
            _w2v.NBestQs(_test_2, bestd, bestw);
            AssertArray.AreAlmostEqual(expectedd, bestd);
            AssertArray.AreEqual(expectedw, bestw);
        }

        [Fact]
        public void TestQuicksortBatch()
        {
            // quicksort batch is not implemented yet
        }

        [Fact]
        public void TestCompatibilityQuicksortBatchSingle()
        {
            // quicksort batch is not implemented yet
        }

        [Fact]
        public void TestCompatibilityQuicksortBatchParallel()
        {
            // quicksort batch is not implemented yet
        }

        [Fact]
        public void TestCompatibilityQuicksortSort()
        {
            // test that quicksort single and sort single have same results

            var test = NN.Random.Normal(0f, 0.1f, 4, 15);
            int neighbors = 5;
            var bestd_1 = new float[neighbors];
            var bestd_2 = new float[neighbors];
            var bestw_1 = new int[neighbors];
            var bestw_2 = new int[neighbors];
            for (int i = 0; i < test.Shape[1]; i++)
            {
                _w2v.NBestQs(test[Slicer._, i], bestd_1, bestw_1);
                _w2v.NBest(test[Slicer._, i], bestd_2, bestw_2);
                AssertArray.AreAlmostEqual(bestd_1, bestd_2);
                AssertArray.AreEqual(bestw_1, bestw_2);
            }
        }
    }
}
