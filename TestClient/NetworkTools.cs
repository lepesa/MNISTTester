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
