﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace MNISTTester
{
    public partial class MNISTTester
    {
        private MNIST mnistData;
        private IMNISTTest testNetwork;
        private int labelsCount;
        
        public MNIST MnistData { get { return mnistData; }}
        public IMNISTTest TestNetwork { get { return testNetwork; } }
        public int LabelsCount { get { return labelsCount; } }

        byte[][] testData;
        byte[] testLabels;

        /// <summary>
        /// Ladataan dll-tiedosto ja sen sisältämä interface.
        /// </summary>
        /// <param name="dllName">Dll tiedoston nimi</param>
        /// <param name="typeName">Luokan nimi, joka dll-tiedostosta ladataan</param>
        public void LoadTestNetwork(string dllName, string typeName)
        {
            Assembly assembly = Assembly.LoadFrom(dllName);
            System.Type type = assembly.GetType(typeName);
            Object o = Activator.CreateInstance(type);
            testNetwork = (o as IMNISTTest);

            mnistData = new MNIST();
        }

        /// <summary>
        /// Tutkii kuinka hyvin neuroverkko tunnistaa numerot. Käytään 10k testiaineisto läpi.
        /// </summary>
        /// <returns>Oikeiden osumien määrän</returns>
        public int GetResults()
        {
            testData = mnistData.TestData;
            testLabels = mnistData.TestLabels;

            int rightNumber = 0;
            labelsCount = testLabels.Length;
            int labelInd = 0;
            for (int i = 0; i<labelsCount - 1; i++)
            {
                // Feedforwardi neuroverkkossa
                labelInd = testNetwork.GetNumber(testData[i]);
                if (labelInd == testLabels[i])
                {
                    rightNumber++;
                }
                // Operaatio keskeytetty
                if( labelInd == -1 )
                {
                    return 0;
                }
            }
            return rightNumber;
        }

        /// <summary>
        /// Keskeytetään operaatio neuroverkon laskennassa.
        /// </summary>
        public void StopOperation()
        {
            if( testNetwork!= null)
            {
                testNetwork.SetStopFlag(true);
            }
        }

        /// <summary>
        /// Jos operaatio on aiemmin keskeytetty, niin sallitaan taas neuroverkon laskenta.
        /// </summary>
        public void StartOperation()
        {
            if (testNetwork != null)
            {
                testNetwork.SetStopFlag(false);
            }
        }

    }
}