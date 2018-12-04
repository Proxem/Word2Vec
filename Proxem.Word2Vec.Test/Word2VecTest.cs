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
using System.IO;
using System.Linq;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Proxem.BlasNet;
using Proxem.NumNet;
using Proxem.NumNet.Single;
using Proxem.NumNet.Double;
using static Proxem.NumNet.Slicer;

namespace Proxem.Word2Vec.Test
{
    [TestClass]
    public class Word2VecTest
    {
        [TestMethod]
        public void CanDumpVectors()
        {
            int vocSize = 100, vecSize = 10;
            var vectors = NN.Random.Uniform(-1f, 1f, vocSize, vecSize);
            var words = Enumerable.Range(0, vocSize).Select(i => i.ToString()).ToArray();
            var w2v = new Word2Vec(words, vectors);

            w2v.SaveBinary(Path.GetTempFileName());
        }

        [TestMethod]
        public void CanReadDumpedVectors()
        {
            int vocSize = 100, vecSize = 10;
            var vectors = NN.Random.Uniform(-1f, 1f, vocSize, vecSize);
            var words = Enumerable.Range(0, vocSize).Select(i => i.ToString()).ToArray();
            var w2v = new Word2Vec(words, vectors);
            var path = Path.GetTempFileName();

            w2v.SaveBinary(path);
            var w2vLoaded = Word2Vec.LoadBinary(path, normalize: false);

            AssertArray.AreEqual(vectors, w2vLoaded.Vectors);
            AssertArray.AreEqual(words, w2vLoaded.Text);
        }

        [Ignore]
        [TestMethod]
        public void CanLoadOldVectors()
        {
            // TODO: test on small w2v
        }

        public static Array<float> PseudoInv(Array<float> a)
        {
            // https://en.wikipedia.org/wiki/Moore–Penrose_pseudoinverse
            // http://vene.ro/blog/inverses-pseudoinverses-numerical-issues-speed-symmetry.html

            // https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/296030
            // dgelss can do the job with one input your matrix, the other the unit matrix
            // http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=2&t=160

            var m = a.Shape[0];
            var n = a.Shape[1];

            /* Compute SVD */
            var k = Math.Min(m, n);
            var s = new float[k];
            var u = NN.Zeros<float>(m, m);
            var vt = NN.Zeros<float>(n, n);
            var copy = (float[])a.Values.Clone(); // if (jobu != 'O' && jobv != 'O') a is destroyed by dgesdv (https://software.intel.com/en-us/node/521150)
            var superb = new float[k - 1];
            Lapack.gesvd('A', 'A', m, n, copy, n, s, u.Values, m, vt.Values, n, superb);

            var invSigma = NN.Zeros<float>(n, m);
            invSigma[Range(0, k), Range(0, k)] = NN.Diag(1 / NN.Array(s));

            var pseudoInv = vt.T.Dot(invSigma).Dot(u.T);
            return pseudoInv;
        }

        public static Array<float> PowerMethod(Array<float> a)
        {
            // https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse

            // init A(0) = (A*A + dI)-1.A*
            var d = 1e-6f;
            var result = a.T.Dot(a) + d * NN.Eye(a.Shape[1]);
            Lapack.Inverse(result.Values, result.Shape[0]);
            result = result.Dot(a.T);

            // iterate: A(i+1) = 2A(i) - A(i).A.A(i)
            for (int i = 0; i < 2; i++)
            {
                result = 2 * result - result.Dot(a).Dot(result);
            }
            return result;
        }

        [Ignore]
        [TestMethod]
        public void TestPseudoInverse()
        {
            var path = @"C:\Users\joc\AppData\Local\ProtoStudio\Banque\embeddings.bin";
            var words = Word2Vec.LoadBinary(path, normalize: true).Vectors/*[_, Until(100)]*/;
            //var pseudoInv = PseudoInv(words);
            var pseudoInv = PowerMethod(words);

            // when embeddings have linearly independent dimensions
            AssertArray.AreAlmostEqual(NN.Eye(words.Shape[1]), pseudoInv.Dot(words), 1e-6f, 1e-6f);

            // least probable: words are NOT linearly idependent
            //AssertArray.AreAlmostEqual(NN.Eye(words.Shape[0]), words.Dot(pseudoInv), 1e-3f, 1e-5f);

            if (words.Shape[0] <= 1000) // otherwise too long
            {
                AssertArray.AreAlmostEqual(words, words.Dot(pseudoInv).Dot(words), 1e-6f, 1e-6f);
            }
            AssertArray.AreAlmostEqual(pseudoInv, pseudoInv.Dot(words).Dot(pseudoInv), 1e-6f, 1e-6f);
        }
    }
}
