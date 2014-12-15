using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestClient
{
    public static class NetworkTools
    {
        /// <summary>
        /// Palauttaa satunnaisluvun välillä -1 ... 1
        /// </summary>
        /// <param name="nrg">Satunnaislukugeneraattoriolio</param>
        /// <returns>Satunnaisluku</returns>
        public static double InitValue(Random nrg)
        {
            return (nrg.NextDouble() * 2 - 1);
        }

        /// <summary>
        /// Simppeli normaalijakaumaa noudattava satunnaislukugeneraattori. Algoritmina 
        /// Box-Muller transformaatio ( http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform )
        /// </summary>
        /// <param name="nrg">Satunnaislukugeneraattoriolio</param>
        /// <param name="mean">keskiarvo</param>
        /// <param name="stdev">keskihajonta</param>
        /// <returns>Satunnaisluvun gaussin käyrään</returns>
        public static double GaussianRandom(Random nrg, double mean = 0, double stdev = 1)
        {
            double u1 = nrg.NextDouble(); //these are uniform(0,1) random doubles
            double u2 = nrg.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return (mean + stdev * randStdNormal); //random normal(mean,stdDev^2)
        }
    }
}
