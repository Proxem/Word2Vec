/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Linq;
using System.Threading.Tasks;
using Proxem.NumNet;
using Proxem.NumNet.Single;

namespace Proxem.Word2Vec
{
    public static class Sorting
    {
        /// <summary>
        /// Quickselect is used to narrow down the search for k-nearest neighbors 
        /// to a small number of candidates. It uses a variant of Quicksort where only
        /// the left part of the array is discarded after each iteration
        /// see https://en.wikipedia.org/wiki/Quicksort
        /// and https://en.wikipedia.org/wiki/Quickselect
        /// </summary>
        /// <param name="scores"> list of scores from which we want to extract the nbests highest values</param>
        /// <param name="mapping"> allows keep track of permutations when pivoting </param>
        /// <param name="nbests"> number of highest values wanted </param>
        /// <param name="initPivot"> first pivot that will be used when partitionning </param>
        /// <param name="threshold"> 
        /// quickselect stops when it find the (unordered) threshold * nbests
        /// best candidates. Only those candidates can then be explored to find the 
        /// orderered nbests.
        /// </param>
        /// <returns> the remaining number of candidates to analyse </returns>
        public static int Quickselect(float[] scores, int[] mapping, int nbests,
            float initPivot = -1, int threshold = 3)
        {
            var left = 0;
            var pivotValue = initPivot <= 0 ? FindPivot(scores, left) : initPivot;
            while (left < scores.Length - threshold * nbests)
            {
                left = Partition(scores, mapping, left, pivotValue);
                pivotValue = FindPivot(scores, left);
            }
            return left;
        }

        /// <summary>
        /// Heapsort on a single dimensional array.
        /// Find the k highest values and indices in the given array and return them in the given bestd and bestw
        /// </summary>
        /// <param name="scores"> array to sort </param>
        /// <param name="bestd"> array to store highest values of score </param>
        /// <param name="bestw"> array to store indices associated with highest values of score </param>
        public static void Heapsort(float[] scores, float[] bestd, int[] bestw)
        {
            var mapping = Enumerable.Range(0, scores.Length).ToArray();
            Heapify(scores, mapping);
            var iter = 0;
            while (iter < bestd.Length)
            {
                bestw[iter] = mapping[0];
                bestd[iter] = scores[0];
                scores[0] = scores[scores.Length - iter - 1];
                mapping[0] = mapping[scores.Length - iter - 1];
                iter++;
                SiftDown(scores, mapping, 0, scores.Length - iter - 1);
            }
        }

        /// <summary>
        /// Parallel version of heapsort for a 2-dimensional array. Heapsort is made on each column in parallel.
        /// The k highest values and indices of each column are returned in the given bestd and bestw
        /// </summary>
        /// <param name="scores"> 2-d array to sort by columns </param>
        /// <param name="bestd"> array to store the highest values of each column </param>
        /// <param name="bestw">array to store the indices associated with the highest values of each column </param>
        public static void HeapsortParallel(Array<float> scores, float[][] bestd, int[][] bestw)
        {
            Parallel.ForEach(Enumerable.Range(0, scores.Shape[1]), i => 
            {
                HeapsortCol(scores, i, bestd[i], bestw[i]);
            });
        }

        /// <summary>
        /// Heapsort on a 2-dimensional array. heapsort is made on each column.
        /// The k highest values and indices of each column are returned in the given bestd and bestw
        /// </summary>
        /// <param name="scores"> 2-d array to sort by columns </param>
        /// <param name="bestd"> array to store the highest values of each column </param>
        /// <param name="bestw">array to store the indices associated with the highest values of each column </param>
        public static void Heapsort(Array<float> scores, float[][] bestd, int[][] bestw)
        {
            for (int i = 0; i < scores.Shape[1]; i++)
            {
                HeapsortCol(scores, i, bestd[i], bestw[i]);
            }
        }

        /// <summary>
        /// Performs heapsort on the given column of the array.
        /// </summary>
        /// <param name="scores"> array to sort </param>
        /// <param name="col"> column number </param>
        /// <param name="bestd"> array to store the highest values of each column </param>
        /// <param name="bestw">array to store the indices associated with the highest values of each column </param>
        private static void HeapsortCol(Array<float> scores, int col, float[] bestd, int[] bestw)
        {
            var mapping = Enumerable.Range(0, scores.Shape[0]).ToArray();
            Heapify(scores, col, mapping);
            var iter = 0;
            while (iter < bestd.Length)
            {
                bestw[iter] = mapping[0];
                bestd[iter] = (float)scores[0, col];
                scores[0, col] = (float)scores[scores.Shape[0] - iter - 1, col];
                mapping[0] = mapping[scores.Shape[0] - iter - 1];
                iter++;
                SiftDown(scores, col, mapping, 0, scores.Shape[0] - iter - 1);
            }
        }

        /// <summary>
        /// Simple sort on the given one dimensional array of scores.
        /// </summary>
        /// <param name="scores"> array to sort </param>
        /// <param name="bestd"> array to store highest values of score </param>
        /// <param name="bestw"> array to store indices associated with highest values of score </param>
        public static void Sort(float[] scores, float[] bestd, int[] bestw)
        {
            var l = scores.Length;
            var mapping = Enumerable.Range(0, l).ToArray();
            Array.Sort(scores, mapping);
            for (int i = 1; i < bestd.Length + 1; i++)
            {
                bestd[i - 1] = scores[l - i];
                bestw[i - 1] = mapping[l - i];
            }
        }

        public static void Sort(Array<float> scores, float[][] bestd, int[][] bestw)
        {
            for (int i = 0; i < scores.Shape[1]; i++)
            {
                SortCol(scores, i, bestd[i], bestw[i]);
            }
        }

        public static void SortParallel(Array<float> scores, float[][] bestd, int[][] bestw)
        {
            Parallel.ForEach(Enumerable.Range(0, scores.Shape[1]), i =>
            {
                SortCol(scores, i, bestd[i], bestw[i]);
            });
        }

        private static void SortCol(Array<float> scores, int i, float[] bestd, int[] bestw)
        {
            //var mapping = Enumerable.Range(0, scores.Shape[0]).ToArray();
            //Array.Sort(scores[Slicer._, i], mapping);
            //for (int j = 0; j < bestd.Length; j++)
            //{
            //    bestd[j] = (float)scores[j, i];
            //    bestw[j] = mapping[j];
            //}
            throw new NotImplementedException();
        }

        /// <summary>
        /// performs one split of quicksort given a pivot value and the position of the previous pivot
        /// </summary>
        /// <param name="scores"> array to sort </param>
        /// <param name="mapping"> mapping of indices </param>
        /// <param name="left"> previous position of the pivot = starting position of current pivoting operation</param>
        /// <param name="pivotValue"> value of the new pivot </param>
        /// <returns></returns>
        private static int Partition(float[] scores, int[] mapping, int left, float pivotValue)
        {
            var storeIndex = left;
            for (int i = left; i < scores.Length; i++)
            {
                if (scores[i] < pivotValue)
                {
                    var temp = scores[storeIndex];
                    scores[storeIndex] = scores[i];
                    scores[i] = temp;
                    var temp2 = mapping[storeIndex];
                    mapping[storeIndex] = mapping[i];
                    mapping[i] = temp2;
                    storeIndex++;
                }
            }
            return storeIndex;
        }

        /// <summary>
        /// Heuristic to find a good new pivot given previous position of the pivot
        /// </summary>
        /// <param name="scores"> array of scores to sort </param>
        /// <param name="left"> previous position of the pivot </param>
        /// <returns></returns>
        private static float FindPivot(float[] scores, int left)
        {
            // mean value of first + last + random value
            var centerIndex = NN.Random.NextInt(scores.Length - left);
            return 1f / 3 * (scores[left] + scores[centerIndex + left] + scores[scores.Length - 1]);
        }

        /// <summary>
        /// turns scores into a max-heap, performs the same operations on mapping to keep track of the indices
        /// </summary>
        /// <param name="scores"> array to turn into a max-heap </param>
        /// <param name="mapping"> array of indices to keep track </param>
        private static void Heapify(float[] scores, int[] mapping)
        {
            int count = scores.Length - 1;
            int start = Parent(count);
            while (start >= 0)
            {
                SiftDown(scores, mapping, start, count);
                start--;
            }
        }

        /// <summary>
        /// turns the given column of scores into a max-heap, performs the same operations on mapping to keep track of the indices
        /// </summary>
        /// <param name="scores"> array to turn into a max-heap </param>
        /// <param name="i"> column number </param>
        /// <param name="mapping"> array of indices to keep track </param>
        public static void Heapify(Array<float> scores, int i, int[] mapping)
        {
            int count = scores.Shape[0] - 1;
            int start = Parent(count);
            while (start >= 0)
            {
                SiftDown(scores, i, mapping, start, count);
                start--;
            }
        }

        /// <summary>
        /// siftdown operation for repairing an heap. pops down values that doesn't respect the heap property
        /// </summary>
        /// <param name="scores"> heap to repair </param>
        /// <param name="mapping"> array to keep track of indices </param>
        /// <param name="start"> starting index to repair </param>
        /// <param name="end"> end of the heap </param>
        private static void SiftDown(float[] scores, int[] mapping, int start, int end)
        {
            var root = start;
            var lchild = LeftChild(root);
            while (lchild  <= end)
            {
                var child = lchild;
                var swap = root;
                if (scores[swap] < scores[child])
                {
                    swap = child;
                }
                if (child + 1 <= end && scores[swap] < scores[child + 1])
                {
                    swap = child + 1; 
                }
                if (swap == root)
                    return;
                else
                {
                    var temp = scores[root];
                    scores[root] = scores[swap];
                    scores[swap] = temp;
                    var temp2 = mapping[root];
                    mapping[root] = mapping[swap];
                    mapping[swap] = temp2;
                    root = swap;
                    lchild = LeftChild(root);
                }
            }
        }

        /// <summary>
        /// siftdown operation for repairing an heap. pops down values that doesn't respect the heap property
        /// </summary>
        /// <param name="scores"> heap to repair </param>
        /// <param name="i"> column number </param>
        /// <param name="mapping"> array to keep track of indices </param>
        /// <param name="start"> starting index to repair </param>
        /// <param name="end"> end of the heap </param>
        private static void SiftDown(Array<float> scores, int i, int[] mapping, int start, int end)
        {
            var root = start;
            var lchild = LeftChild(root);
            while (lchild <= end)
            {
                var child = lchild;
                var swap = root;
                if ((float)scores[swap, i] < (float)scores[child, i])
                {
                    swap = child;
                }
                if (child + 1 <= end && (float)scores[swap, i] < (float)scores[child + 1, i])
                {
                    swap = child + 1;
                }
                if (swap == root)
                    return;
                else
                {
                    var temp = (float)scores[root, i];
                    scores[root, i] = (float)scores[swap, i];
                    scores[swap, i] = temp;
                    var temp2 = mapping[root];
                    mapping[root] = mapping[swap];
                    mapping[swap] = temp2;
                    root = swap;
                    lchild = LeftChild(root);
                }
            }
        }

        private static int Parent(int pos)
        {
            return (pos % 2) + (pos / 2) - 1;
        }

        private static int LeftChild(int pos)
        {
            return pos * 2 + 1;
        }

        private static int RightChild(int pos)
        {
            return (pos + 1) * 2;
        }
    }

}
