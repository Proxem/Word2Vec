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
using System.IO;
using System.Text;

namespace Proxem.Word2Vec
{
    public static class BinaryReaderExtensions
    {
        static byte[] buffer = new byte[1000];

        /// <summary>
        /// Reads a string delimited by a char
        /// </summary>
        public static string ReadString(this BinaryReader reader, char delimiter = '\0')
        {
            byte c;
            int i = 0;
            while ((c = reader.ReadByte()) != delimiter)
            {
                if (i >= buffer.Length) Array.Resize(ref buffer, buffer.Length * 2);
                buffer[i++] = c;
            }
            return Encoding.UTF8.GetString(buffer, 0, i);
        }

        /// <summary>
        /// Write a delimited string.
        /// </summary>
        public static void WriteString(this BinaryWriter writer, string s, char delimiter = '\0')
        {
            foreach (var c in s)
                writer.Write(c);
            writer.Write(delimiter);
        }
    }
}
