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
using System.Runtime.CompilerServices;

namespace TestClient
{
    public class Network
    {
        
        public enum ActivateFunction { InputLayer, Sigmoid, Tanh, Softmax, Softplus, ReLU };
        public enum CostFunction { Quadratic, CrossEntropy };

        public readonly Layer[] layers;

        public double parLearningRate = 0.1;
        public double parMomentum = 0;
        public double parweightDecay = 0;

        // Sigmoidin / Tanh:n vaatimat pointterit
        public CostFunction costFunctionType = CostFunction.Quadratic;

        /// <summary>
        /// Tallentaa verkon hyper parametrit säilöön. Näitä arvoja ei kuitenkaan käytetä suoraan metodeissa.
        /// </summary>
        /// <param name="learningRate">Learning rate / epsilon</param>
        /// <param name="momentum">Momentum / alpha</param>
        /// <param name="lambda">Weight decay / lambda</param>
        public void SetHyperParameters(double learningRate, double momentum, double weightDecay)
        {
            parLearningRate = learningRate;
            parMomentum = momentum;
            parweightDecay = weightDecay;
        }
                
        /// <summary>
        /// Alustaa verkon: tekee layerit, asettaa aktivointi ja hintafunktion.
        /// Lisäksi asettaa hyperparametrit.
        /// </summary>
        /// <param name="layerSizes">Layerien nodemäärät</param>
        /// <param name="funcs">Aktivointifunktiot per layer</param>
        /// <param name="cost">Hintafunktio output-layerille</param>
        /// <param name="learningRate">Learning rate / epsilon</param>
        /// <param name="momentum">Momentum / alpha</param>
        /// <param name="lambda">Weight decay / lambda</param>
        public Network(int[] layerSizes, ActivateFunction[] funcs, CostFunction cost, double learningRate, double momentum, double weightDecay) : this(layerSizes, funcs, cost)
        {
            SetHyperParameters(learningRate, momentum, weightDecay);
        }

        /// <summary>
        /// Alustaa verkon: tekee layerit, asettaa aktivointi ja hintafunktion
        /// </summary>
        /// <param name="layerSizes">Layerien nodemäärät</param>
        /// <param name="funcs">Aktivointifunktiot per layer</param>
        /// <param name="cost">Hintafunktio output-layerille</param>
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
                if( funcs[i] == ActivateFunction.Softmax)
                {
                    layers[i].ActivateFunc = ActivateSoftmax;
                    layers[i].DerivateFunc = DerivateSoftmax;
                }
                if (funcs[i] == ActivateFunction.Softplus)
                {
                    layers[i].ActivateFunc = ActivateSoftplus;
                    layers[i].DerivateFunc = DerivateSoftplus;
                }

                if (funcs[i] == ActivateFunction.ReLU)
                {
                    layers[i].ActivateFunc = ActivateReLU;
                    layers[i].DerivateFunc = DerivateReLU;
                }
            }
        }

        /// <summary>
        /// Aseta uudet satunnaiset painotukset kaikille layereille
        /// </summary>
        public void Reset()
        {
            foreach (var layer in layers)
            {
                layer.Reset();
            }
        }
        /// <summary>
        /// Aseta uudet gaussin käyrän mukaisesti satunnaiset painotukset kaikille layereille
        /// </summary>
        public void ResetGaussian()
        {
            foreach (var layer in layers)
            {
                layer.ResetGaussian();
            }
        }

        /// <summary>
        /// Feedforward: syötä input arvot ja saa output-arvot vastaukseksi kun ollaan
        /// laskettu ne verkon läpi. Layereiden output-arvot muuttuvat.
        /// </summary> 
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
                    // optimointi
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

                    currentLayer.outputValue[j] = currentLayer.ActivateFunc(output);
                }  

                if( currentLayer.activateFunctionType == ActivateFunction.Softmax)
                {
                    CalculateSoftmaxBuffer(currentLayer.outputValue);
                }


            }
        }
        /// <summary>
        /// Toteuttaa Rectified Linear Unitin aktivointifunktion.
        /// f(x) = max(0, x)
        /// </summary>
        /// <param name="value">Arvo</param>
        /// <returns>Arvo ReLUn jälkeen</returns>
        private static double ActivateReLU(double value)
        {
            if (value < 0)
            {
                return 0;
            }
            else
            {
                return value;
            }
        }

        /// <summary>
        /// Toteuttaa Rectified Linear Unitin derivaatan:
        /// koska f(x) = max(0,x), niin derivaatta on joko 0 tai 1.
        /// </summary>
        /// <param name="value">Arvo</param>
        /// <returns>Arvo RerLUn derivaatan jälkeen</returns>
        private static double DerivateReLU(double value)
        {
            if (value > 0)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }

        /// <summary>
        /// Toteuttaa Softplus funktion f(x) = ln(1+e^x)
        /// </summary>
        /// <param name="value">Arvo</param>
        /// <returns>Arvo Softplussin jälkeen</returns>
        private static double ActivateSoftplus(double value)
        {
            return Math.Log(1 + Math.Exp(value));
        }

        /// <summary>
        /// Toteuttaa Softplus funktion f(x) = ln(1+e^x) derivaatan, joka on
        /// f'(x) = e^x/(e^x+1) = 1/(1+e^-x)
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        private static double DerivateSoftplus(double value)
        {
            return (1 / (1 + Math.Exp(-1 * value)));
        }


        /// <summary>
        /// Toteuttaa sigmoid-funktion. S(t)= 1 / (1+e^-t)
        /// </summary>
        /// <param name="value">Arvo</param>
        /// <returns>Arvo sigmoidin jälkeen</returns>
        private static double ActivateSigmoid(double value)
        {
            return (1.0 / (1 + Math.Exp(-1.0 * value)));
        }

        /// <summary>
        /// Toteuttaa sigmoid-funktion. S(t)= 1 / (1+e^-t) derivaatan, joka on
        /// S'(t) = t * (1-t)
        /// </summary>
        /// <param name="value">Arvo</param>
        /// <returns>Arvo Sigmoidin derivaatan jälkeen</returns>
        private static double DerivateSigmoid(double value)
        {
            return value * (1 - value);
        }

        /// <summary>
        /// Toteuttaa tanh-funktion (Hyperbolic tangent). Tähän käytetään kirjastofunktiota.
        /// </summary>
        /// <param name="value">Arvo</param>
        /// <returns>Arvo TanH:n jälkeen</returns>
        private static double ActivateTanh(double value)
        {
            return Math.Tanh(value);
        }

        /// <summary>
        /// Toteuttaa tanh-funktion derivaatan, joka on
        /// f'(x) = 1 - (Tanh(x)^2)
        /// </summary>
        /// <param name="value">Arvo</param>
        /// <returns>Arvo TanH derivaatan jälkeen</returns>
        private static double DerivateTanh(double value)
        {
            return (1.0 - Math.Pow(ActivateTanh(value), 2.0));
        }

        /// <summary>
        /// Laskee softmaxin annetuille arvoille.
        /// Ensiksi lasketaan syötearvojen e^(xi) summa ja sen jälkeen jokainen syöte jaetaan summalla. Tätä käytetään laskiessa
        /// Feedforwardissa.
        /// </summary>
        /// <param name="values">Layerin output-arvot</param>
        private static void CalculateSoftmaxBuffer(double[] values)
        {
            double sum = 0;
            for (int i = 0; i < values.Length; i++)
            {
                sum += Math.Exp(values[i]);
            }
            for(int i=0; i < values.Length; i++)
            {
                values[i] =  Math.Exp(values[i]) / sum;
            }

        }
        /// <summary>
        /// Palauttaa softmaxin halutulle arvolle. Tämä lasketaan myöhemmin CalculateSoftmaxBufferissa, 
        /// koska tarvitaan tietoon _kaikkien_ output-arvot. Joten palautetaan vain saatu arvo, jotta  
        /// FeedForwardin logiikka pysyy yksinkertaisena.
        /// </summary>
        /// <param name="value">Arvo</param>
        /// <returns>Saatu input arvo</returns>
        private static double ActivateSoftmax(double value)
        {
            return value;
            
        }

        /// <summary>
        /// Softmaxin derivaatta on sama kuin sigmoidilla, eli 
        /// S'(t) = t * (1-t)
        /// </summary>
        /// <param name="value">Arvo</param>
        /// <returns>Arvo Softmaxin derivaatan jälkeen</returns>
        private static double DerivateSoftmax(double value)
        {
            return value * (1 - value);
        }

        /// <summary>
        /// Tyhjentää Minibatchin käyttämien muuttujien arvot.
        /// Huomaa että gradients tyhjennetään backprop-metodissa.
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
            }
        }
        /// <summary>
        /// Laskee yhdelle minibatchin stepille virheen. Tämä virhe lisätään gradienttiin, jota käytetään myöhemmin loopin loputtua.
        /// </summary>
        /// <param name="desiredResult">Halutut verkon tulokset</param>
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
                for (k = currentLayerCount - 1; k >= 0; k--)
                {
                    // Paino kerrotaan 1.0:lla, koska bias
                    currentLayer.errorValueTemp[k] = 1.0 * currentLayer.errorValue[k];
                    
                    currentLayer.gradients[biasIndex][k] += currentLayer.errorValueTemp[k];
                }

                // Normaalien painotuksien päivitys
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

        /// <summary>
        /// Päivitetään minibatchin painotuksien arvot. Tämä vastaa Backpropagation -metodia Online -oppimisessa.
        /// </summary>
        /// <param name="learningRate">Oppimiskerroin</param>
        /// <param name="momentum">Momenttikerroin</param>
        /// <param name="lambda">Weight decay kerroin</param>
        /// <param name="batchSize">Minibatchin koko</param>
        /// <param name="trainingSize">Opetusaineiston koko</param>
        public void UpdateMinibatchValues(double learningRate, double momentum, double lambda, int batchSize, int trainingSize)
        {
            Layer currentLayer;      
            Layer previousLayer;     
            int i;
            int j;
            int k;
            int currentLayerCount;
            int prevLayerCount;


            // Lasketaan weight decay arvo valmiiksi
            double l2reg = 1 - (learningRate * (lambda/ trainingSize));

            // Opetuskerroin ja momentti pitää jakaa batchin koolla, koska kumulatiivinen summa
            learningRate = learningRate / batchSize;
            momentum = momentum / batchSize;

            int biasIndex;
            double weightDiff;
            // Nyt on virheet tiedossa. Päivitetään verkon painotukset lopusta alkuun.
            for (i = layers.Length - 1; i > 0; i--)
            {
                currentLayer = layers[i];
                previousLayer = layers[i - 1];

                biasIndex = currentLayer.weights.Length - 1;

                currentLayerCount = currentLayer.neuronCount;
                prevLayerCount = previousLayer.neuronCount;

                // Bias  päivitys
                for (k = currentLayerCount - 1; k >= 0; k--)
                {
                    // Saadaam delta-arvo  derivoitu virhearvosta kerrottuna oppimisarvolla [0..1]. Laitetaan tämä talteen ja lisäksi lisätään se painoarvoon                   
                    weightDiff  = learningRate * currentLayer.gradients[biasIndex][k];

                    // Lisätään delta, nyt paino on w(t+1)
                    currentLayer.weights[biasIndex][k] += weightDiff;

                    // Lisätään painoarvoon momenttiarvo kerrottuna edellisen kerran delta-arvolla
                    currentLayer.weights[biasIndex][k] += momentum * currentLayer.prevWeightDiffs[biasIndex][k];

                    // Asetetaan delta-arvo talteen seuraavaa laskukertaa varten
                    currentLayer.prevWeightDiffs[biasIndex][k] = weightDiff;

                    // Nollataan summagradientti
                    currentLayer.gradients[biasIndex][k] = 0;
                }

                // Normaalien painotuksien päivitys
                for (j = 0; j < prevLayerCount; j++)
                {
                    for (k = 0; k < currentLayerCount; k++)
                    {
                        // Lasketaan ei-bias arvot
                        weightDiff = learningRate * currentLayer.gradients[j][k];

                        // Lasketaan uusi paino, josta on otettu weight decay, lisätään ero ja momentti
                        currentLayer.weights[j][k] = l2reg * currentLayer.weights[j][k] + weightDiff + momentum * currentLayer.prevWeightDiffs[j][k];
                                       
                        // Vanha delta talteen seuraavaa kierrosta varten
                        currentLayer.prevWeightDiffs[j][k] = weightDiff;

                        // Nollataan summagradientti
                        currentLayer.gradients[j][k] = 0;
                    }
                }
            }
        }

        /// <summary>
        /// Opetetaan verkkoa. Käytännössä siis muutetaan verkon painotuksia odotettujen ja laskettujen arvojen perusteella.
        /// Ennen tätä on yleensä feedforward suoritettu, jotta on jotain arvoja mitä opettaa.
        /// </summary>
        /// <param name="desiredResult">Verkon haluttu tulos</param>
        /// <param name="learningRate">Oppimiskerroin</param>
        /// <param name="momentum">Momenttikerroin</param>
        /// <param name="lambda">Weight decay kerroin</param>
        /// <param name="trainingSize">Opetusaineiston koko</param>

        public virtual void Backpropagation(double[] desiredResult, double learningRate, double momentum, double lambda, int trainingSize)
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

            Layer currentLayer;
            Layer previousLayer;

            int biasIndex;
            double weightDiff;

            double l2reg = 1 - (learningRate * (lambda / trainingSize));

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
                for (k = currentLayerCount - 1; k >= 0; k--)
                {
                    // Saadaan delta-arvo  derivoitu virhearvosta kerrottuna oppimisarvolla [0..1]. Laitetaan tämä talteen ja lisäksi lisätään se painoarvoon
                    weightDiff = currentLayer.errorValueTemp[k] = learningRate * currentLayer.errorValue[k];

                    // Lisätään delta, nyt paino on w(t+1)
                    currentLayer.weights[biasIndex][k] += weightDiff + momentum * currentLayer.prevWeightDiffs[biasIndex][k];

                    // Asetetaan delta-arvo talteen seuraavaa laskukertaa varten
                    currentLayer.prevWeightDiffs[biasIndex][k] = weightDiff;
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

                        // Asetetaan delta-arvo talteen seuraavaa laskukertaa varten
                        currentLayer.prevWeightDiffs[j][k] = weightDiff;
                    }
                }
            }
        }
    }
}
