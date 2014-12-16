using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MNISTTester;

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
            int bestCount = 0;
            TimeSpan totalTime = new TimeSpan();

            int epoch = 1;
            Stopwatch sw;
            mnistTester.TestNetwork.InitDatas(mnistTester.MnistData.ImageData, mnistTester.MnistData.ImageLabels);
            mnistTester.StartOperation();
            timer.Start();
            while (!token.IsCancellationRequested && epoch <= maxEpochs)
            {
                AddLogLine("Starting training epoch: " + epoch);
                sw = Stopwatch.StartNew();

                mnistTester.TestNetwork.TrainEpoch();

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
