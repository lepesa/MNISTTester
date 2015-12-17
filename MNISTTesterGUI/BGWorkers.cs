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
using System.Diagnostics;
using System.Threading;

namespace MNISTLoaderGUI
{
    public partial class MainWindow
    {
        /// <summary>
        /// Ajaa neuroverkon opetusloopin omassa threadissaan, jotta UI ei häiriinny.
        /// </summary>
        /// <param name="obj">CancellationToken</param>
        public void bgWorkerRunEpoches(object obj)
        {
            CancellationToken token = (CancellationToken)obj;

            int maxEpochs = MaxEpochs;
            int miniBatchSize = MiniBatchSize;
            int bestCount = 0;
            TimeSpan totalTime = new TimeSpan();

            int epoch = 1;
            Stopwatch sw;
            mnistTester.TestNetwork.InitDatas(mnistTester.MnistData.ImageData, mnistTester.MnistData.ImageLabels);
            mnistTester.StartOperation();
            timer.Start();

            if( miniBatchSize == 1 )
            {
                AddLogLine("Using stochastics gradient descent" );
            } else
            {
                AddLogLine("Using mini-batch gradient descent. Batch size: " + miniBatchSize);
            }

            AddLogLine(mnistTester.TestNetwork.GetNetworkInfo());


            while (!token.IsCancellationRequested && epoch <= maxEpochs)
            {
                AddLogLine("Starting training epoch: " + epoch);
                sw = Stopwatch.StartNew();

                if (miniBatchSize == 1)
                {
                    mnistTester.TestNetwork.TrainEpoch();
                }
                else
                {
                    mnistTester.TestNetwork.TrainEpochMiniBatch(miniBatchSize);
                }

                if (!token.IsCancellationRequested)
                {
                    totalTime += sw.Elapsed;
                    AddLogLine(sw.Elapsed + ": epoch trained.");

                    int rightNumber = mnistTester.GetResults();
                    int labelsCount = mnistTester.LabelsCount;
                    AddLogLine(sw.Elapsed + ": result: " + rightNumber + " / " + labelsCount + ".");

                    if (rightNumber > bestCount)
                    {
                        bestCount = rightNumber;
                    }

                    epoch++;
                }
                else
                {
                    AddLogLine("Cancelled.");
                }
            }
            cdEvent.Signal();
            timer.Stop();
            
            AddLogLine("Test thread ended.");
            AddLogLine(totalTime + " was totaltime. " + (epoch-1) + " loops, best result was " + bestCount);
        }

        /// <summary>
        /// Päivittää progressbaria kerran sekunnissa, kun laskentathreadia ajetaan.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>

        private void timer_UpdateProgressBar(object sender, System.EventArgs e)
        {

            if (mnistTester.TestNetwork != null)
            {
                int percentage = mnistTester.TestNetwork.GetWorkPercentage();
                Dispatcher.BeginInvoke(new Action(delegate()
                {
                    pbStatus.Value = percentage;
                }));
            }
        }
    }
}
