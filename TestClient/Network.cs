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
using System.Runtime.CompilerServices;

namespace TestClient
{
    public class Network
    {
        
        public enum ActivateFunction { InputLayer, Sigmoid, Tanh };
        public enum CostFunction { Quadratic, CrossEntropy };

        public readonly Layer[] layers;

        // Sigmoidin / Tanh:n vaatimat pointterit
     

        public CostFunction costFunctionType = CostFunction.Quadratic;
     
        // Verkon alustus. Saadaan tietoon verkon koko, aktivointifunktio ja maksufunktio
        public Network(int[] layerSizes, ActivateFunction[] funcs, CostFunction cost)
        {

            layers = new Layer[layerSizes.Length];

            costFunctionType = cost;
            

            for (int i = 0; i < layerSizes.Length; i++)
            {
                if (i == 0)
                {
                    // input layer
                    layers[i] = new Layer(layerSizes[i]);
                }
                else
                {
                    // hidden / output
                    layers[i] = new Layer(layerSizes[i], layerSizes[i - 1]);
                }

                layers[i].activateFunctionType = funcs[i];

                // Asetetaan oikeat funktiot feedforwardia/back propagationia varten
                if (funcs[i] == ActivateFunction.Sigmoid)
                {
                    layers[i].ActivateFunc = ActivateSigmoid;
                    layers[i].DerivateFunc = DerivateSigmoid;
                }
                if (funcs[i] == ActivateFunction.Tanh)
                {
                    layers[i].ActivateFunc = ActivateTanh;
                    layers[i].DerivateFunc = DerivateTanh;
                }

            }
           

        }
        // Aseta uudet satunnaiset painotukset kaikille layereille
        public void Reset()
        {
            foreach (var layer in layers)
            {
                layer.Reset();
            }
        }
        // Aseta uudet gaussin käyrän mukaisesti satunnaiset painotukset kaikille layereille
        public void ResetGaussian()
        {
            foreach (var layer in layers)
            {
                layer.ResetGaussian();
            }
        }


        // Feedforward: syötä input arvot ja saa output-arvot vastaukseksi kun ollaan 
        // laskettu ne verkon läpi. Verkon tila ei muutu
        public void FeedForward()
        {
            Layer currentLayer;
            Layer prevLayer;

            double output;
            double output2;
            double output3;
            double output4;
            int prevLength;
            int neur;
            int neur2;
            int j;

            for (int i = 1; i < layers.Length; i++)
            {

                currentLayer = layers[i];
                prevLayer = layers[i - 1];
                prevLength = prevLayer.outputValue.Length;
                for (j = currentLayer.neuronCount-1; j >= 0;  j--)
                {
                    // TODO: Dot product
                    // Calculate Oneur * Wneur*J + ... + ( 1 * Bias)
                    output = output2 = output3 = output4 = 0;
                   
                    neur2 = prevLength % 4;
                    neur = prevLength - 1;

                    switch (neur2)
                    {
                        case 1:
                            output3 += prevLayer.outputValue[neur] * currentLayer.weights[neur--][j];
                            break;
                        case 2:
                            output2 += prevLayer.outputValue[neur] * currentLayer.weights[neur--][j];
                            output3 += prevLayer.outputValue[neur] * currentLayer.weights[neur--][j];
                            break;
                        case 3:
                            output2 += prevLayer.outputValue[neur] * currentLayer.weights[neur--][j];
                            output2 += prevLayer.outputValue[neur] * currentLayer.weights[neur--][j];
                            output3 += prevLayer.outputValue[neur] * currentLayer.weights[neur--][j];
                            break;
                        default:
                            break;
                    }
                    
                    while (neur>=3)
                    {
                        output += prevLayer.outputValue[neur] * currentLayer.weights[neur--][j];
                        output2 += prevLayer.outputValue[neur] * currentLayer.weights[neur--][j];
                        output2 += prevLayer.outputValue[neur] * currentLayer.weights[neur--][j];
                        output3 += prevLayer.outputValue[neur] * currentLayer.weights[neur--][j];
                    }
                    output += output2 + output3 + output4;
                    // sigmoid: 1.0 / (1 + Math.exp(-1.0 * d));
                    //                   layers[i].outputValue[j] = 1.0 / ( 1.0 + Math.Exp( -1.0 * output ));
                    // tanh
                    //layers[i].outputValue[j] = (Math.Exp(output * 2.0) - 1.0) / (Math.Exp(output * 2.0) + 1.0);

                    currentLayer.outputValue[j] = currentLayer.ActivateFunc(output);
                }  
            }
        }

        // Sigmoid -funktio feedforwardia varten
        private static double ActivateSigmoid(double value)
        {
            return (1.0 / (1 + Math.Exp(-1.0 * value)));
        }

        // Sigmoid -funktio back propagationia varten
        private static double DerivateSigmoid(double value)
        {
            return value * (1 - value);
        }

        // Tanh -funktio feedforwardia varten
        private static double ActivateTanh(double value)
        {
            return Math.Tanh(value);
        }

        // Tanh -funktio back propagationia varten
        private static double DerivateTanh(double value)
        {
            return (1.0 - Math.Pow(ActivateTanh(value), 2.0));
        }


        /// <summary>
        /// Tyhjentää Minibatchin käyttämien muuttujien arvot.
        /// </summary>
        public void ClearMiniBatchValues()
        {
            

            for (int i = 1; i < layers.Length; i++)
            {
                Layer clearLayer = layers[i];
                for (int j = clearLayer.neuronCount - 1; j >= 0; j--)
                {
                    clearLayer.errorValue[j] = 0;
                }

            }/*
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].ResetGradients();
            }*/
        }

        public void CalculateMiniBatchError(double[] desiredResult)
        {
            // Lasketaan odotettujen arvojen ja todellisten arvojen välinen virhe: output layer
            Layer outputLayer = layers[layers.Length - 1];
            double outputValue;
            int i;
            int j;
            int currentLayerCount;
            int prevLayerCount;

            if (costFunctionType == CostFunction.Quadratic)
            {
                for (i = outputLayer.neuronCount - 1; i >= 0; i--)
                {
                    // MSE. Huomaa että derivaatta pitää olla oikea, riippuen funktiosta
                    outputValue = outputLayer.outputValue[i];
                    outputLayer.errorValue[i] = (desiredResult[i] - outputValue) * outputLayer.DerivateFunc(outputValue);
                }
            }
            else
            {
                for (i = outputLayer.neuronCount - 1; i >= 0; i--)
                {
                    // MCEE
                    outputLayer.errorValue[i] = (desiredResult[i] - outputLayer.outputValue[i]);
                }
            }

            Layer hiddenLayer;

            // Lasketaan odotettujen arvojen ja todellisten arvojen välinen virhe: hidden layers

            int k;
            for (i = layers.Length - 2; i > 0; i--)
            {
                hiddenLayer = layers[i];
                outputLayer = layers[i + 1];
                prevLayerCount = hiddenLayer.neuronCount;
                currentLayerCount = outputLayer.neuronCount;

                for (j = prevLayerCount - 1; j >= 0; j--)
                {
                    outputValue = 0;

                    for (k = currentLayerCount - 1; k >= 0; k--)
                    {
                        outputValue += outputLayer.weights[j][k] * outputLayer.errorValue[k];
                    }

                    // sigmoid derivate, sigmoid(x) * (1-sigmoid(x))
                    // tanh derviate: (1-tanh(x)^2)

                    hiddenLayer.errorValue[j] = hiddenLayer.DerivateFunc(hiddenLayer.outputValue[j]) * outputValue;

                }
            }

            Layer currentLayer;      // layers[i]
            Layer previousLayer;     // layers[i-1]

            int biasIndex;
            double prevOutputValue;
            // Nyt on virheet tiedossa. Päivitetään verkon gradientit lopusta alkuun.
            for (i = layers.Length - 1; i > 0; i--)
            {

                currentLayer = layers[i];
                previousLayer = layers[i - 1];

                // calculate biases
                biasIndex = currentLayer.weights.Length - 1;

                currentLayerCount = currentLayer.neuronCount;
                prevLayerCount = previousLayer.neuronCount;

                // Bias  päivitys
                for (j = currentLayerCount - 1; j >= 0; j--)
                {
                    // Paino kerrotaan 1.0:lla, koska bias
                    currentLayer.errorValueTemp[j] = 1.0 * currentLayer.errorValue[j];
                    
                    currentLayer.gradients[biasIndex][j] += currentLayer.errorValueTemp[j];
                }

                // Normaalien painotuksein päivitys
                for (j = 0; j < prevLayerCount; j++)
                {
                    prevOutputValue = previousLayer.outputValue[j];
                    for (k = 0; k < currentLayerCount; k++)
                    {
                        // Lasketaan ei-bias arvot. errorValueTemp[x] on tässä 1.0 * errorValue[x]. Tämä on saatu laskettua jo biassien laskemisessa.

                        // Lisätään delta, nyt paino on w(t+1)
                        currentLayer.gradients[j][k] += currentLayer.errorValueTemp[k] * prevOutputValue;    
                    }
                }
            }

        }

        public void UpdateMinibatchValues(double learningRate, double momentum, double lambda, int batchSize, int trainingSize)
        {
            Layer currentLayer;      // layers[i]
            Layer previousLayer;     // layers[i-1]
            int i;
            int j;
            int k;
            int currentLayerCount;
            int prevLayerCount;


            
            double l2reg = 1 - (learningRate * (lambda/ trainingSize));

            learningRate = learningRate / batchSize;
            momentum = momentum / batchSize;

            int biasIndex;
            double weightDiff;
            // Nyt on virheet tiedossa. Päivitetään verkon painotukset lopusta alkuun.
            for (i = layers.Length - 1; i > 0; i--)
            {

                currentLayer = layers[i];
                previousLayer = layers[i - 1];

                // calculate biases
                biasIndex = currentLayer.weights.Length - 1;

                currentLayerCount = currentLayer.neuronCount;
                prevLayerCount = previousLayer.neuronCount;

                // Bias  päivitys
                for (j = currentLayerCount - 1; j >= 0; j--)
                {
                    // Saadaam delta-arvo  derivoitu virhearvosta kerrottuna oppimisarvolla [0..1]. Laitetaan tämä talteen ja lisäksi lisätään se painoarvoon                   
                    weightDiff = currentLayer.errorValueTemp[j] = learningRate * currentLayer.errorValue[j];

                    // Lisätään delta, nyt paino on w(t+1)
                    currentLayer.weights[biasIndex][j] += weightDiff;

                    // Lisätään painoarvoon momenttiarvo kerrottuna edellisen kerran delta-arvolla
                    currentLayer.weights[biasIndex][j] += momentum * currentLayer.prevWeightDiffs[biasIndex][j];

                    // Asetetaan delta-arvo talteen seuraavaa laskukertaa varten
                    currentLayer.prevWeightDiffs[biasIndex][j] = weightDiff;

                    // Nollataan summagradientti
                    currentLayer.gradients[biasIndex][j] = 0;
                }

                // Normaalien painotuksein päivitys
                for (j = 0; j < prevLayerCount; j++)
                {
                    for (k = 0; k < currentLayerCount; k++)
                    {
                        // Lasketaan ei-bias arvot. errorValueTemp[x] on learninRate * errorValue[x]. 
                        weightDiff = learningRate * currentLayer.gradients[j][k];

                        // Lasketaan uusi paino, josta on otettu weight decay, lisätään ero ja momentti
                        currentLayer.weights[j][k] = l2reg * currentLayer.weights[j][k] + weightDiff + momentum * currentLayer.prevWeightDiffs[j][k];
                                       
                        // Vanha delta talteen seuraavaa kierrosta varten
                        currentLayer.prevWeightDiffs[j][k] = weightDiff;

                        // Nollataan gradienttisumma
                        currentLayer.gradients[j][k] = 0;
                    }
                }
            }
        }


         // Opetetaan verkkoa. Käytännössä siis muutetaan verkon painotuksia odotettujen ja laskettujen arvojen perusteella.
         // Ennen tätä on yleensä feedforward suoritettu, jotta on jotain arvoja mitä opettaa.
         public virtual void Backpropagation(double[] desiredResult, double learningRate, double momentum, double lambda, int learningSize)
        { 
            Layer outputLayer = layers[layers.Length - 1];
            double outputValue;
            int i;
            int j;
            int currentLayerCount;
            int prevLayerCount;

            // Lasketaan odotettujen arvojen ja todellisten arvojen välinen virhe: output layer
            
            if (costFunctionType == CostFunction.Quadratic)
            {
                for (i = outputLayer.neuronCount-1 ; i >= 0; i--)
                {
                    // MSE. Huomaa että derivaatta pitää olla oikea, riippuen funktiosta
                    outputValue = outputLayer.outputValue[i];
                    outputLayer.errorValue[i] = (desiredResult[i] - outputValue) * outputLayer.DerivateFunc(outputValue);
                }
            }
            else
            {
                for (i = outputLayer.neuronCount -1; i >= 0; i--)
                {
                    // MCEE
                    outputLayer.errorValue[i] = (desiredResult[i] - outputLayer.outputValue[i]);
                }
            }

            Layer hiddenLayer;

            // Lasketaan odotettujen arvojen ja todellisten arvojen välinen virhe: hidden layers
                        
            int k;
            for (i = layers.Length - 2; i > 0; i--)
            {
                hiddenLayer = layers[i];
                outputLayer = layers[i + 1];
                prevLayerCount = hiddenLayer.neuronCount;
                currentLayerCount = outputLayer.neuronCount;

                for (j = prevLayerCount - 1; j >= 0; j--)
                {
                    outputValue = 0;

                    for (k = currentLayerCount - 1; k >= 0; k--)
                    {
                        outputValue += outputLayer.weights[j][k] * outputLayer.errorValue[k];
                    }

                    hiddenLayer.errorValue[j] = hiddenLayer.DerivateFunc(hiddenLayer.outputValue[j]) * outputValue;

                }
            }

            Layer currentLayer;      // layers[i]
            Layer previousLayer;     // layers[i-1]

            int biasIndex;
            double weightDiff;

            double l2reg = 1 - (learningRate * (lambda / learningSize));

            // Nyt on virheet tiedossa. Päivitetään verkon painotukset lopusta alkuun.
            for (i = layers.Length - 1; i > 0; i--)
            {

                currentLayer = layers[i];
                previousLayer = layers[i - 1];

                // calculate biases
                biasIndex = currentLayer.weights.Length - 1;

                currentLayerCount = currentLayer.neuronCount;
                prevLayerCount = previousLayer.neuronCount;

                // Bias  päivitys
                for (j = currentLayerCount - 1; j >= 0; j--)
                {
                    // Saadaan delta-arvo  derivoitu virhearvosta kerrottuna oppimisarvolla [0..1]. Laitetaan tämä talteen ja lisäksi lisätään se painoarvoon
                    
                    weightDiff = currentLayer.errorValueTemp[j] = learningRate * currentLayer.errorValue[j];

                    // Lisätään delta, nyt paino on w(t+1)
                    currentLayer.weights[biasIndex][j] += weightDiff + momentum * currentLayer.prevWeightDiffs[biasIndex][j];

                    // Asetetaan delta-arvo talteen seuraavaa laskukertaa varten
                    currentLayer.prevWeightDiffs[biasIndex][j] = weightDiff;
                }

                // Normaalien painotuksein päivitys
                for (j = 0; j < prevLayerCount; j++)
                {

                    for (k = 0; k < currentLayerCount; k++)
                    {
                        // Lasketaan ei-bias arvot. errorValueTemp[x] on learninRate * errorValue[x]. Tämä on saatu laskettua jo biassien laskemisessa.
                        weightDiff = currentLayer.errorValueTemp[k] * previousLayer.outputValue[j] ;

                        // Lasketaan uusi paino, josta on otettu weight decay, lisätään ero ja momentti
                        currentLayer.weights[j][k] = l2reg * currentLayer.weights[j][k] + weightDiff + momentum * currentLayer.prevWeightDiffs[j][k];

                        currentLayer.prevWeightDiffs[j][k] = weightDiff;

                    }
                }
            }
        }
    }
}
