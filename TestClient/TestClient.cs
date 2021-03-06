﻿/*
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
using MNISTTester;
using System.Text;

namespace TestClient
{
    /// <summary>
    /// Esimerkkitoteutus, jota ohjelma kutsuu. Käyttää simppeliä neuroverkkoa, jolla pääsee 95-98% tarkkuuteen, riippuen neuronien määrästä.
    /// Tehty esimerkiksi, joten rakenne hieman ontuu. Yritetty nopeusoptimoida enemmän tai vähemmän hyvin.
    /// </summary>
    public class TestClient : IMNISTTest
    {

        private const string VERSION_STRING = "TestClient example v1.2";

        private Network network;
        private int materialIndex;
        private int materialSize;

        private int[] dataIndex;
        private double[][] data;

        private double[] idealValues = null;

        private byte[] _desiredDatas = null;
        private byte[][] _imageDatas = null;

        private bool stopOperation = false;
        private Random nrg = null;

        /// <summary>
        /// Alustetaan neuroverkko.
        /// </summary>
        public TestClient()
        {

            network = new Network(new Layer[] { new Layer(28 * 28, 0.3), new Layer(400, 28*28,0), new Layer(10, 400, 0) }, new Network.ActivateFunction[] { Network.ActivateFunction.InputLayer, Network.ActivateFunction.Sigmoid, Network.ActivateFunction.Tanh }, Network.CostFunction.CrossEntropy, 0.1, 0.0, 0);       
       
            network.ResetGaussian();
            nrg = new Random();
        }

        /// <summary>
        /// Palautetaan versioinfo.
        /// </summary>
        /// <returns>Versioinfo</returns>
        public string GetVersion()
        {
            return VERSION_STRING;
        }

        /// <summary>
        /// Alustetaan opetusmateriaali verkkoa varten.
        /// </summary>
        /// <param name="imageDatas">Kuvatiedostot, kuvia n kpl, 28*28 pikselin kuva</param>
        /// <param name="desiredDatas">Kuvatiedostojen selitteet, kertoo mikä numero kuvassa</param>
        public void InitDatas(byte[][] imageDatas, byte[] desiredDatas)
        {
            materialSize = imageDatas.Length;
            dataIndex = new int[materialSize];

            for (int i = 0; i < materialSize; i++)
            {
                dataIndex[i] = i;
            }
                        
            _desiredDatas = desiredDatas;
            _imageDatas = imageDatas;
            
            int imageSize = imageDatas[0].Length;

            data = new double[materialSize][];

            for (int i = 0; i < materialSize; i++)
            {
                data[i] = new double[imageSize];
                
            }

            Layer outputLayer = network.layers[network.layers.Length - 1];
            int outputSize = outputLayer.neuronCount;
            idealValues = new double[outputSize];
        }

        /// <summary>
        /// Päivitetään kuvadata uusiksi sekoittamalla sen järjestys ja normalisoimalla se.
        /// </summary>
        private void InitializeImageData()
        {
            int newI;
            int tmpI;
            
            // Yritetään satunnaistaa opetusjärjestystä -> parempi oppiminen kun kuvat
            // eivät ole joka kerralla samassa järjestyksessä. Tehokkaampi sekoitus olisi kiva,
            // mutta tämän ajankäyttö on O(n) ja jokaisella kerralla aineisto on melko varmasti
            // eri kohdassa kuin viimeksi.

            for (int i = 0; i < _imageDatas.Length; i++)
            {
                newI = nrg.Next(0, _imageDatas.Length);
                if (i != newI)
                {
                    tmpI = dataIndex[newI];
                    dataIndex[newI] = dataIndex[i];
                    dataIndex[i] = tmpI;
                }
            }


            // Normalisoidaan datat. Nyt ne ovat vielä välillä 0...255, halutaan välille 0...1
            for (int i = 0; i < _imageDatas.Length; i++)
            {
                for (int j = 0; j < _imageDatas[0].Length; j++)
                {
                    data[i][j] = NormalizeImageData(_imageDatas[dataIndex[i]][j], network.layers[1].activateFunctionType);
                }
            }
        }

        /// <summary>
        /// Käy läpi kaikki kuvat ja opettaa/laskee niiden perusteella verkkoa.
        /// Käytetään online stochastic back-propagation, eli lasketaan outputit, saadaan niistä virhe ja päivitetään verkkoa 
        /// jokaisen yhden laskun (kuvan) jälkeen.
        /// </summary>
        public void TrainEpoch()
        {
            // progressbar arvot
            
            materialIndex = 0;
            materialSize = _imageDatas.Length;

            int imageSize = _imageDatas[0].Length;


            InitializeImageData();

            Layer outputLayer = network.layers[network.layers.Length - 1];
            int outputSize = outputLayer.neuronCount;

            double oldIdealValue;
            // Output-arvot edustavat numeroita 0...9. Asetetaan haluttu indeksi ykköseksi.
            if (network.layers[network.layers.Length - 1].activateFunctionType != Network.ActivateFunction.Tanh)
            {
                Array.Clear(idealValues, 0, outputSize);
            }
            else
            {
                for (int j = 0; j < outputSize; j++)
                {
                    idealValues[j] = -1;
                }
            }

            // Mahdolliset dropoutit
            bool dropoutInUse = false;

            for(int i=0; i<network.layers.Length;i++)
            {
                if (network.layers[i].dropOutValue > 0)
                {
                    dropoutInUse = true;
                    network.layers[i].CreateDropOut(false);
                }
            }

            // Opetetaan kuva kerrallaan. Ekana feedforward, sen jälkeen backpropagation.
            // Käytetään kiinteitä arvoja: oppimiskerroin 0.3. momentti on käytössä ja asetettu arvoon 0.1.
            // weight decayta ei ole

            for (int i = 0; i < materialSize; i++,materialIndex++  )
            {
                for (int j = 0; j < imageSize; j++)
                {
                    network.layers[0].outputValue[j] = data[i][j];
                }

                network.FeedForward();

                oldIdealValue = idealValues[_desiredDatas[dataIndex[i]]];
                idealValues[_desiredDatas[dataIndex[i]]] = 1.0f;

                // learning rate, momentum, weight decay/l2 reg
                if (dropoutInUse)
                {
                    network.BackpropagationDropout(idealValues, network.parLearningRate, network.parMomentum, network.parweightDecay, 60000);
                }
                else
                {
                    network.Backpropagation(idealValues, network.parLearningRate, network.parMomentum, network.parweightDecay, 60000);
                } 
                idealValues[_desiredDatas[dataIndex[i]]] = oldIdealValue;
                if ( stopOperation)
                {
                    break;
                }
            }

            // Drop outit pois, jotta testaaminen olisi ok
            for (int i = 0; i < network.layers.Length; i++)
            {
                network.layers[i].ResetDropOut();
            }

            return;            
        }

        /// <summary>
        /// Käy läpi kaikki kuvat ja opettaa/laskee niiden perusteella verkkoa. Tämä tehdään käyttämällä minibatcheja.
        /// ts. verkolle tehdään opetus vasta kun on laskettu virhe batchSize:n ilmaisemalle määrälle elementtejä ja 
        /// tästä otetaan keskiarvo.
        /// </summary>
        public void TrainEpochMiniBatch(int batchSize)
        {
            // progressbar arvot
            materialIndex = 0;
            materialSize = _imageDatas.Length;

            int imageSize = _imageDatas[0].Length;

            InitializeImageData();

            Layer outputLayer = network.layers[network.layers.Length - 1];
            int outputSize = outputLayer.neuronCount;


            double oldIdealValue;
            // Output-arvot edustavat numeroita 0...9. Asetetaan haluttu indeksi ykköseksi.
            if (network.layers[network.layers.Length - 1].activateFunctionType != Network.ActivateFunction.Tanh)
            {
                Array.Clear(idealValues, 0, outputSize);
            }
            else
            {
                for (int j = 0; j < outputSize; j++)
                {
                    idealValues[j] = -1;
                }
            }

            // Mahdolliset dropoutit

            bool dropoutInUse = false;

            for (int i = 0; i < network.layers.Length; i++)
            {
                if (network.layers[i].dropOutValue > 0)
                {
                    dropoutInUse = true;
                    network.layers[i].CreateDropOut(false);
                }
            }



            for (int matIndex = 0; matIndex < materialSize; matIndex+=batchSize, materialIndex+=batchSize)
            {

                network.ClearMiniBatchValues();

                // Laske yhteinen virhe
                for (int batchNro = 0; batchNro < batchSize; batchNro++)
                {
                    for (int j = 0; j < imageSize; j++)
                    {
                        network.layers[0].outputValue[j] = data[matIndex+batchNro][j];
                    }

                    network.FeedForward();

                    // Koska vain yksi arvo kerrallaan on yksi, niin otetaan vanha arvo talteen ja asetetaan myöhemmin takaisin
                    oldIdealValue = idealValues[_desiredDatas[dataIndex[matIndex + batchNro]]];
                    idealValues[_desiredDatas[dataIndex[matIndex+ batchNro]]] = 1.0f;

                    if (dropoutInUse)
                    {
                        network.CalculateMiniBatchErrorDropout(idealValues);
                    } else
                    {
                        network.CalculateMiniBatchError(idealValues);
                    }
                    idealValues[_desiredDatas[dataIndex[matIndex + batchNro]]] = oldIdealValue;

                }

                // learning rate, momentum
                if (dropoutInUse)
                {
                    network.UpdateMinibatchValuesDropout(network.parLearningRate, network.parMomentum, network.parweightDecay, batchSize, 60000);
                } else
                {
                    network.UpdateMinibatchValues(network.parLearningRate, network.parMomentum, network.parweightDecay, batchSize, 60000);
                }
              
                if (stopOperation)
                {
                    break;
                }
            }
            // Drop outit pois, jotta testaaminen olisi ok
            for (int i = 0; i < network.layers.Length; i++)
            {
                network.layers[i].ResetDropOut();
            }

            return;
        }
    

        /// <summary>
        /// Normalisoi arvot välille 0...1 (sigmoid) tai -1...1 (tanh) 
        /// </summary>
        /// <param name="data">Pikselin arvo välillä 0-255</param>
        /// <returns>Normalisoitu arvo</returns>
        private double NormalizeImageData(byte data, Network.ActivateFunction func)
        {
            double temp = (double)data;
            if (func != Network.ActivateFunction.Tanh)
            {
                // väli: 0...1
                return (temp / 255);
            }
            else
            {
                //tanh: väli -1...1
                return (temp / 127.5 - 1.0);
            }
        }

        /// <summary>
        /// Yrittää tunnistaa kuvan esittämän numeron.
        /// </summary>
        /// <param name="numberData">28x28 kuvan datat</param>
        /// <returns>Kuvan esittämän numeron indeksi</returns>
        public int GetNumber(byte[] numberData)
        {
            if( stopOperation)
            {
                return -1;
            }

            for (int j = 0; j < numberData.Length; j++)
            {
                network.layers[0].outputValue[j] = NormalizeImageData(numberData[j], network.layers[1].activateFunctionType);
            }
            // Pelkkä feedforward, ei haluta oppia mitään -> 
            // ei tarvitse muuttaa verkon painotuksia.
            network.FeedForward();

            int numberIndex = -1;
            double maxValue = -9999;

            Layer outputLayer = network.layers[network.layers.Length - 1];

            // Haetaan millä neuronilla on isoin arvo -> se on "oikea" vastaus
            for (int j = 0; j < outputLayer.neuronCount; j++)
            {
                if (outputLayer.outputValue[j] > maxValue)
                {
                    numberIndex = j;
                    maxValue = outputLayer.outputValue[j];
                }
            }

            return numberIndex;
        }

        /// <summary>
        /// Palautetaan prosentti, missä vaiheessa aineiston käsittely on
        /// </summary>
        /// <returns>Prosentti</returns>
        public int GetWorkPercentage()
        {
            if (materialSize > 0)
            {
                return (100 * materialIndex / materialSize);
            }
            return 0;
        }

        /// <summary>
        /// Asettaa stoppilipun true/flase. Ei käytetä lukitusta, vaikka kutsutaankin 
        /// toisesta threadista kuin missä itse laskenta pyörii, koska tämä on ainoa paikka,
        /// missä arvoa muutetaan.
        /// </summary>
        public void SetStopFlag(bool value)
        {
            stopOperation = value;
        }

        /// <summary>
        /// Palauttaa verkon tiedot tekstimuodossa.
        /// </summary>
        /// <returns>Verkon tiedot</returns>
        public string GetNetworkInfo()
        {
            StringBuilder info = new StringBuilder("Network:" + System.Environment.NewLine);
         

            for (int i = 0; i < network.layers.Length; i++)
            {
                if( i == 0)
                {
                    info.Append(String.Format("Input layer, {0} nodes, activation func: {1}, dropout: {2}", network.layers[i].neuronCount, network.layers[i + 1].activateFunctionType, network.layers[i].dropOutValue));
                }
                else if ((i + 1) == network.layers.Length)
                {
                    info.Append(String.Format("Output layer, {0} nodes, Cost func: {1}", network.layers[i].neuronCount, network.costFunctionType));
                }
                else
                {
                    info.Append(String.Format("Hidden layer {2}, {0} nodes, activation func: {1}, dropout: {3}", network.layers[i].neuronCount, network.layers[i + 1].activateFunctionType, i, network.layers[i].dropOutValue));
                }
                info.Append(System.Environment.NewLine);
            }

            info.Append(System.Environment.NewLine + "Hyperparams:" + System.Environment.NewLine);
            info.Append(String.Format("Learning rate: {0}, Momentum: {1}, Lambda/Weight decay: {2}/Training set size", network.parLearningRate, network.parMomentum, network.parweightDecay));
            
            return info.ToString();
        }
    }
}
