/*
   Copyright 2015 Esa Leppänen
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

using System;
using System.IO;
using System.Linq;

namespace MNISTTester
{
    /// <summary>
    /// MNIST kuvien käsittelyyn rutiinit 
    /// </summary>
    public class MNIST
    {
        private byte[][] imageData;
        private byte[] imageLabels;

        private byte[][] imageDataTest;
        private byte[] imageLabelsTest;

        public UInt32 imageCount;
        public UInt32 imageRows;
        public UInt32 imageCols;

        public byte[][] ImageData { get {return imageData;}}
        public byte[] ImageLabels { get {return imageLabels; }}
        public byte[][] TestData { get { return imageDataTest; }} 
        public byte[] TestLabels { get { return imageLabelsTest; }}


        /// <summary>
        /// Lataa kiinteästi määritellyt tiedostot. Nämä pitää löytyä työhakemistosta. Nämä tiedostot löytyvät:
        /// http://yann.lecun.com/exdb/mnist/ 
        /// </summary>
        public void LoadMnistData()
        {
            LoadSet( "train-images.idx3-ubyte", "train-labels.idx1-ubyte", ref imageData, ref imageLabels, 60000 );
            LoadSet( "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", ref imageDataTest, ref imageLabelsTest, 10000 );
        }

        /// <summary>
        /// Lataa annetun kuvatiedoston ja sen labelit annettuihin säiliöihin. Yrittää havaita tavujärjestyksen.
        /// </summary>
        /// <param name="imageFilename">Kuvatiedoston nimi</param>
        /// <param name="labelFilename">Labelitiedoston nimi</param>
        /// <param name="imgData">Objekti, johon säilötään kuvat</param>
        /// <param name="labelData">Objekti, johon säilötään labelit</param>
        /// <param name="imgCount">Kuvien määrä</param>
        private void LoadSet(string imageFilename, string labelFilename, ref byte[][] imgData, ref byte[] labelData, UInt32 imgCount )
        {

            /** TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
            [offset] [type]          [value]          [description]
            0000     32 bit integer  0x00000803(2051) magic number
            0004     32 bit integer  60000            number of images
            0008     32 bit integer  28               number of rows
            0012     32 bit integer  28               number of columns
            0016     unsigned byte   ??               pixel
            0017     unsigned byte   ??               pixel
            ........
            xxxx     unsigned byte   ??               pixel
            */

            using (BinaryReader reader = new BinaryReader(File.Open(imageFilename, FileMode.Open)))
            {
                UInt32 magicNumber = ReadUInt32(reader);
                imageCount = ReadUInt32(reader);
                imageRows = ReadUInt32(reader);
                imageCols = ReadUInt32(reader);

                // Tunniste on 2051, kuvia pitää olla haluttu määrä ja kuvan koon pitää olla 28*28.
                if (magicNumber != 2051 || imageCount != imgCount || imageCols != 28 || imageRows != 28)
                {
                    throw new Exception( "Error when reading images. MagicNumber(2051): " + magicNumber + ", images (" + imgCount + "): " + imageCount + 
                       ", cols(28): " + imageCols + ", rows(28): " + imageRows);
                }

                imgData = new byte[imageCount][];

                int imageSize = (int)(imageRows * imageCols);

                for (int imageNr = 0; imageNr < imageCount; imageNr++)
                {
                    imgData[imageNr] = new byte[imageSize];
                    imgData[imageNr] = reader.ReadBytes(imageSize);
                }
            }

            /*
                * TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
                [offset] [type]          [value]          [description]
                0000     32 bit integer  0x00000801(2049) magic number (MSB first)
                0004     32 bit integer  60000            number of items
                0008     unsigned byte   ??               label
                0009     unsigned byte   ??               label
                ........
                xxxx     unsigned byte   ??               label
            */
            using (BinaryReader reader = new BinaryReader(File.Open(labelFilename, FileMode.Open)))
            {
                UInt32 magicNumber = ReadUInt32(reader);
                UInt32 labelCount = ReadUInt32(reader);
                
                // Tunnisteen pitää olla 2049 ja tunnisteita pitää olla kuvien määrä.
                if (magicNumber != 2049 || labelCount != imageCount)
                {
                    throw new Exception("Error when reading labels. MagicNumber (2049): " + magicNumber + ", labels(" + imageCount +") : " + labelCount);
                }

                labelData = new byte[labelCount];
                reader.Read(labelData, 0, (Int32)labelCount);
            }
        }

        /*
         * Datafile's format is  
         */
        /// <summary>
        /// Lutetaan yksi 32 bittinen luku tiedostosta.
        /// Tiedoston tavujärjestys on BigEndian. Yritetään kääntää järjestelmän haluamaan muotoon tavut.
        /// </summary>
        /// <param name="reader">BinaryReader</param>
        /// <returns>uint32 BinaryReaderista</returns>
        private UInt32 ReadUInt32(BinaryReader reader)
        {
            if (BitConverter.IsLittleEndian)
            {
                return BitConverter.ToUInt32(reader.ReadBytes(4).Reverse().ToArray(), 0);
            }
            else
            {
                return reader.ReadUInt32();
            }
        }
    }
}
