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

using System.Collections.Generic;
using System;
using System.Linq;

namespace Proxem.Word2Vec
{
    // https://en.wikipedia.org/wiki/Vantage-point_tree
    // http://stevehanov.ca/blog/index.php?id=130

    public class VPTree<T>
    {
        private int[] Indices;
        private T[] Items;
        private Ball Root;

        private Func<T, T, double> Distance;    // euclidean distance satisfying triangle inequality

        private class Ball
        {
            public int Index;
            public double Radius;
            public Ball Left;
            public Ball Right;
        }

        public VPTree(T[] items, Func<T, T, double> distance, System.Random rand = null)
        {
            this.Indices = Enumerable.Range(0, items.Length).ToArray();
            this.Items = items;
            this.Distance = distance;
            this.Root = BuildTree(0, items.Length, rand ?? new System.Random(0xBeef));
        }

        private Ball BuildTree(int left, int right, System.Random rand)
        {
            if (right == left) return null;

            var ball = new Ball();
            ball.Index = left;

            if (right - left > 1)
            {
                Swap(Indices, left, rand.Next(left + 1, right));

                int mid = (right + left) / 2;

                // rearrange indices such that
                // - elements before mid are lower than mid
                // - elements after mid are greater than mid
                Partition(Indices, left + 1, right - 1, mid,
                    (item1, item2) => Comparer<double>.Default.Compare(Distance(Items[Indices[left]], Items[item1]), Distance(Items[Indices[left]], Items[item2])));

                ball.Radius = this.Distance(Items[Indices[left]], Items[Indices[mid]]);

                ball.Left = BuildTree(left + 1, mid, rand);
                ball.Right = BuildTree(mid, right, rand);
            }

            return ball;
        }

        public int Search(T target, (int index, double dist)[] hits)
        {
            int count = Search(Root, target, hits, 0);

            // restore original indices
            for (int i = 0; i < count; i++)
            {
                var hit = hits[i];
                hits[i] = (Indices[hit.index], hit.dist);
            }

            return count;
        }

        public int Search(T target, int[] bestw, double[] bestd)
        {
            int count = Search(Root, target, bestw, bestd, 0);

            // restore original indices
            for (int i = 0; i < count; i++)
            {
                bestw[i] = Indices[bestw[i]];
            }

            return count;
        }

        private int Search(Ball ball, T target, (int index, double dist)[] hits, int count)
        {
            if (ball == null) return count;

            double dist = this.Distance(Items[Indices[ball.Index]], target);

            if (count < hits.Length || dist < hits[count - 1].dist)
            {
                // We found entry with shorter distance
                var pos = 0;
                while (pos < count)
                {
                    if (dist < hits[pos].dist)
                    {
                        Array.Copy(hits, pos, hits, pos + 1, Math.Min(count, hits.Length - 1) - pos);
                        break;
                    }
                    ++pos;
                }

                if (pos == hits.Length) --pos;
                hits[pos] = (ball.Index, dist);

                if (count < hits.Length) ++count;
            }

            if (ball.Left == null && ball.Right == null) return count;

            var maxDist = hits[count - 1].dist;
            if (dist < ball.Radius)
            {
                if (dist - maxDist <= ball.Radius)
                {
                    count = Search(ball.Left, target, hits, count);
                }

                if (dist + maxDist >= ball.Radius)
                {
                    count = Search(ball.Right, target, hits, count);
                }
            }
            else
            {
                if (dist + maxDist >= ball.Radius)
                {
                    count = Search(ball.Right, target, hits, count);
                }

                if (dist - maxDist <= ball.Radius)
                {
                    count = Search(ball.Left, target, hits, count);
                }
            }

            return count;
        }

        private int Search(Ball ball, T target, int[] bestw, double[] bestd, int count)
        {
            if (ball == null) return count;

            double dist = this.Distance(Items[Indices[ball.Index]], target);

            if (count < bestd.Length || dist < bestd[count - 1])
            {
                // We found entry with shorter distance
                var pos = 0;
                while (pos < count)
                {
                    if (dist < bestd[pos])
                    {
                        Array.Copy(bestd, pos, bestd, pos + 1, Math.Min(count, bestd.Length - 1) - pos);
                        Array.Copy(bestw, pos, bestw, pos + 1, Math.Min(count, bestw.Length - 1) - pos);
                        break;
                    }
                    ++pos;
                }

                if (pos == bestd.Length) --pos;
                bestd[pos] = dist;
                bestw[pos] = ball.Index;

                if (count < bestd.Length) ++count;
            }

            if (ball.Left == null && ball.Right == null) return count;

            var maxDist = bestd[count - 1];
            if (dist < ball.Radius)
            {
                if (dist - maxDist <= ball.Radius)
                {
                    count = Search(ball.Left, target, bestw, bestd, count);
                }

                if (dist + maxDist >= ball.Radius)
                {
                    count = Search(ball.Right, target, bestw, bestd, count);
                }
            }
            else
            {
                if (dist + maxDist >= ball.Radius)
                {
                    count = Search(ball.Right, target, bestw, bestd, count);
                }

                if (dist - maxDist <= ball.Radius)
                {
                    count = Search(ball.Left, target, bestw, bestd, count);
                }
            }
            return count;
        }

        private static void Swap(int[] indices, int i, int j)
        {
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Rearranges the list in such a way such that the element at the nth position is the one which should be at that position
        // if we sort the list.
        // It does not sort the list, just that all the elements, which precede the nth element are not greater than it, and all the
        // elements which succeed it are not less than it.
        // see https://www.geeksforgeeks.org/stdnth_element-in-cpp/
        private static void Partition(int[] indices, int from, int to, int nth, Comparison<int> comparison)
        {
            // if from == to we reached the kth element
            while (from < to)
            {
                int left = from, right = to;
                var pivot = indices[(left + right) / 2];

                // stop if the left and right meet
                while (left < right)
                {
                    if (comparison(indices[left], pivot) > -1)
                    { // put large values at the end
                        Swap(indices, right, left);
                        right--;
                    }
                    else
                    { // the value is smaller than the pivot, skip
                        left++;
                    }
                }

                // if we stepped up (left++) we need to step one down
                if (comparison(indices[left], pivot) > 0)
                {
                    left--;
                }

                // the left pointer is on the end of the first k elements
                if (nth <= left)
                {
                    to = left;
                }
                else
                {
                    from = left + 1;
                }
            }
        }
    }
}