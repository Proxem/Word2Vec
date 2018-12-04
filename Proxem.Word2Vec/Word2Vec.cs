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
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Proxem.NumNet;
using Proxem.NumNet.Single;

namespace Proxem.Word2Vec
{
    public class Word2Vec
    {
        public const string UNK = "<UNK>";
        public readonly string[] Text;
        public readonly Array<float> Vectors;
        public readonly Dictionary<string, int> Index;
        private Dictionary<string, int> _indexIgnoreCase;

        //public Word2Vec(int vocSize, int vecSize)
        //{
        //    Text = new string[vocSize];
        //    Index = new Dictionary<string, int>(vocSize);
        //    Vectors = NN.Zeros(vocSize, vecSize);
        //}

        public Word2Vec(Dictionary<string, int> index, Array<float> embeddings)
        {
            embeddings.AssertOfDim(2);
            Vectors = embeddings;
            Index = index;

            Text = new string[embeddings.Shape[0]];
            foreach (var kv in index)
                Text[kv.Value] = kv.Key;
            BuildIndexIgnoreCase();
        }

        public Word2Vec(string[] words, Array<float> vectors)
        {
            var vocSize = words.Length;
            vectors.AssertOfDim(2);
            vectors.AssertOfShape(words.Length, vectors.Shape[1]);

            Vectors = vectors;
            Text = words;

            Index = new Dictionary<string, int>(vocSize);
            for (int i = 0; i < vocSize; ++i)
                Index[words[i]] = i;
            BuildIndexIgnoreCase();
        }

        private void BuildIndexIgnoreCase()
        {
            _indexIgnoreCase = new Dictionary<string, int>(StringComparer.InvariantCultureIgnoreCase);
            foreach (var kvp in Index)
            {
                _indexIgnoreCase[kvp.Key] = kvp.Value;
            }
        }

        public void SaveBinary(string filename, int maxCount = int.MaxValue, string prefix = "")
        {
            var count = Math.Min(maxCount, Text.Length);

            using (var writer = File.CreateText(filename))
            {
                writer.WriteLine($"{count} {VectorSize}");
                for (int n = 0; n < count; n++)
                {
                    writer.Write($"{Text[n]} ");
                    writer.Flush();
                    for (int i = 0; i < VectorSize; i++)
                    {
                        var buffer = BitConverter.GetBytes((float)Vectors.Item[n, i]);
                        writer.BaseStream.Write(buffer, 0, buffer.Length);
                    }
                    writer.Write("\n");
                }

                Trace.WriteLine($"Saved {count} words of size {VectorSize} to {filename}.");
            }
        }

        public static Word2Vec LoadBinary(string filename, bool normalize = true, bool addUnk = false, string prefix = "", Encoding encoding = null)
        {
            int maxcount = int.MaxValue;
            var pos = filename.IndexOf(';');
            if (pos != -1)
            {
                maxcount = int.Parse(filename.Substring(pos + 1));
                filename = filename.Substring(0, pos);
            }
            using (var stream = File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var reader = new BinaryReader(stream, encoding ?? Encoding.GetEncoding(1252)))
            {
                var parts = reader.ReadString('\n').Split(' ');
                var extra = addUnk ? 1 : 0;
                var count = Math.Min(maxcount, int.Parse(parts[0])) + extra;
                var size = int.Parse(parts[1]);

                //var result = new Word2Vec(count, size);
                var vectors = new Array<float>(count, size);
                var index = new Dictionary<string, int>();

                int off = 0, n = 0;
                var buffer = vectors.Values;
                if (addUnk)
                {
                    index[UNK] = n;
                    off += size;
                    ++n;
                }

                for (; n < count; n++)
                {
                    var word = prefix + reader.ReadString(' ').Trim();
                    //result.Text[n] = word;
                    for (int i = 0; i < size; i++)
                    {
                        buffer[off] = reader.ReadSingle();
                        ++off;
                    }

                    // allows a word to appear twice when loading the embeddings,
                    // the vector returned by the index would be the first one
                    if (index.ContainsKey(word))
                        Trace.WriteLine($"Warning: word {word} appears at least twice in the embeddings {filename}");
                    else
                        index.Add(word, n);

                    if (normalize)
                        Normalize(vectors[n]);
                    //Console.WriteLine("{0}\t{1}", result.Text[b], result.MRaw[b].Norm());
                }

                Trace.WriteLine(string.Format("Loaded {0} words of size {1} from {2}.", count, size, filename));
                return new Word2Vec(index, vectors);
            }
        }

        public static Word2Vec LoadText(string filename, bool normalize = true, bool addUnk = false, string prefix = "")
        {
            int maxcount = int.MaxValue;
            var pos = filename.IndexOf(';');
            if (pos != -1)
            {
                maxcount = int.Parse(filename.Substring(pos + 1));
                filename = filename.Substring(0, pos);
            }
            using (var f = new StreamReader(File.OpenRead(filename)))
            {
                var parts = f.ReadLine().Split(' ');
                var extra = addUnk ? 1 : 0;
                var count = Math.Min(maxcount, int.Parse(parts[0])) + extra;
                var size = int.Parse(parts[1]);

                //var result = new Word2Vec(count, size);
                var vectors = new Array<float>(count, size);
                var index = new Dictionary<string, int>();

                int off = 0, n = 0;
                var buffer = vectors.Values;
                if (addUnk)
                {
                    index[UNK] = n;
                    off += size;
                    ++n;
                }

                for (; n < count; n++)
                {
                    var line = f.ReadLine();
                    parts = line.Split('\t');
                    if (parts.Length == 1) parts = line.Split(' ');

                    var word = prefix + parts[0];
                    //result.Text[n] = word;
                    for (int i = 0; i < size; i++)
                    {
                        buffer[off] = float.Parse(parts[i + 1], CultureInfo.InvariantCulture);
                        ++off;
                    }

                    // allows a word to appear twice when loading the embeddings,
                    // the vector returned by the index would be the first one
                    if (index.ContainsKey(word))
                        Trace.WriteLine($"Warning: word {word} appears at least twice in the embeddings {filename}");
                    else
                        index.Add(word, n);

                    if (normalize)
                        Normalize(vectors[n]);
                    //Console.WriteLine("{0}\t{1}", result.Text[b], result.MRaw[b].Norm());
                }

                Trace.WriteLine(string.Format("Loaded {0} words of size {1} from {2}.", count, size, filename));
                return new Word2Vec(index, vectors);
            }
        }

        public int VectorSize => Vectors.Shape[1];
        public int Count => Index.Count;

        public Array<float> this[int wordIndex]
        {
            get
            {
                return Vectors[wordIndex];
            }
            set
            {
                Vectors[wordIndex] = value;
            }
        }

        public Array<float> this[string word]
        {
            get
            {
                if (!Index.ContainsKey(word) && Index.ContainsKey(UNK))
                    word = UNK;
                return Vectors[Index[word]];
            }
            set
            {
                Vectors[Index[word]] = value;
            }
        }

        public int FindWord(string word)
        {
            word = word.Replace(' ', '_');
            int result;
            if (Index.TryGetValue(word, out result)) return result;
            if (_indexIgnoreCase.TryGetValue(word, out result)) return result;

            //Trace.WriteLine($"Word {word} was not found in dictionary");
            return -1;
        }

        public static void Normalize(Array<float> v)
        {
            var norm = NN.Norm(v);
            if (norm != 0) v.Div(norm, result: v);
        }

        public virtual void NBest(Array<float> mb, float[] bestd, int[] bestw, Func<string, bool> filter = null)
        {
            for (int i = 0; i < bestd.Length; i++) bestd[i] = float.NegativeInfinity;
            for (int c = 0; c < this.Text.Length; c++)
            {
                if (filter != null && !filter(this.Text[c])) continue;
                float dist = mb.VectorDot(this.Vectors[c]);

                for (int a = 0; a < bestd.Length; a++)
                {
                    if (dist > bestd[a])
                    {
                        for (int d = bestd.Length - 1; d > a; d--)
                        {
                            bestd[d] = bestd[d - 1];
                            bestw[d] = bestw[d - 1];
                        }
                        bestd[a] = dist;
                        bestw[a] = c;
                        break;
                    }
                }
            }
        }

        public void NBestQs(Array<float> mb, float[] bestd, int[] bestw, float initPivot = 0, int threshold = 3)
        {
            var scores = NN.Dot(Vectors, mb);
            var mapping = Enumerable.Range(0, Text.Length).ToArray();
            int k = Sorting.Quickselect(scores.Values, mapping, bestd.Length, initPivot, threshold);

            for (int i = 0; i < bestd.Length; i++) bestd[i] = float.NegativeInfinity;
            for (int c = k; c < Text.Length; c++)
            {
                for (int a = 0; a < bestd.Length; a++)
                {
                    var dist = scores.Values[c];
                    if (dist > bestd[a])
                    {
                        for (int d = bestd.Length - 1; d > a; d--)
                        {
                            bestd[d] = bestd[d - 1];
                            bestw[d] = bestw[d - 1];
                        }
                        bestd[a] = dist;
                        bestw[a] = mapping[c];
                        break;
                    }
                }
            }
        }

        public void NBestHeap(Array<float> mb, float[] bestd, int[] bestw)
        {
            var scores = NN.Dot(Vectors, mb);
            var mapping = Enumerable.Range(0, Text.Length).ToArray();
            Sorting.Heapsort(scores.Values, mapping, bestd, bestw);
        }

        private VPTree<Array<float>> _vpTree;
        public void NBestVPTree(Array<float> mb, double[] bestd, int[] bestw)
        {
            if (_vpTree == null)
            {
                _vpTree = new VPTree<Array<float>>(ArrayVectors(), DotProduct);
            }
            _vpTree.Search(mb, bestw, bestd);
        }

        private double DotProduct(Array<float> vec1, Array<float> vec2)
        {
            return (double)NN.Dot(vec1, vec2);
        }

        private Array<float>[] ArrayVectors()
        {
            var vectors = new Array<float>[Text.Length];
            for (int i = 0; i < vectors.Length; i++)
            {
                vectors[i] = this[Text[i]];
            }
            return vectors;
        }
    }
}
