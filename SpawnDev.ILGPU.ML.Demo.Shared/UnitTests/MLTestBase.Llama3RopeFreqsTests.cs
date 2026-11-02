using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Text;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // 75 generated facts for the long-context case, 9 characters each: colour index, island index (hex), steps (3 digits), year (4 digits). Python's
    // random.Random(7) drew them (scratch script ollama_ref.py, 2026-11-02); the prompt token count below proves the text rebuilt here is the one llama.cpp read.
    private const string LighthouseFacts =
        "5224219460131418045906919093008418916116318036032918113a3611929093391881030631922242541816191971923213371926350891920190" +
        "7019383738819166c20018997519318432b16418004829318677435117981825418225229018870a0791922553951869792731797142821950103991" +
        "8597423718680722118231707018354216618816e29418002724519204e11018904b2521871631171801221581948302881930241841781263131874" +
        "523931911073881923662441880173641882030741833720961867010401925280911873011461937623641844592261901112891899771991801212" +
        "1518477d39418210331018722b31817864a08618465e12518713831719085a15419363c16218823330519065b0541787471721829572181873130921" +
        "8387321218327935217807e37418681d37818106c14219022636518651c24218986b0831820";

    // llama.cpp's own top-8 log-probabilities at each step of the two reference answers (Ollama 0.34.4 `logprobs` / `top_logprobs`, same request as the
    // answers): per step "id:logprob" x8, steps by ';'. Ids from Meta's tiktoken tokenizer rebuilt from the GGUF vocabulary; one control-token candidate
    // Ollama returns without its bytes (short step 17, at -8.44) is left out. Script: scratch logprob_fixture.py, 2026-11-02.
    private const string LlamaCppTop8_short =
        "791:-0.1111,8586:-2.2742,20215:-7.0084,3947:-7.3915,32:-7.9298,5159:-8.5658,16352:-10.2630,40:-10.9291;2380:-0.0034,6156:-5.6941,3116:-13.1412,1403:-1" +
        "3.2900,5144:-13.3235,220:-13.7433,14853:-14.2491,4320:-14.3397;6156:-0.0000,1925:-10.5827,6913:-11.3650,8776:-12.5090,16188:-13.0061,26150:-13.4491,99" +
        "99:-15.0190,6682:-15.1179;8146:-0.0000,1933:-12.3644,27230:-13.2607,9718:-13.7398,21712:-14.4373,34770:-16.3988,28296:-17.0041,320:-17.3627;527:-0.000" +
        "1,315:-9.6525,430:-10.7824,358:-11.4871,304:-11.9840,11:-12.1586,1473:-12.5377,320:-13.4630;1473:-0.0011,25:-7.3898,3816:-7.8555,2579:-9.7734,8868:-13" +
        ".8922,11383:-14.2823,6437:-14.6174,512:-14.6880;16:-0.0003,6161:-8.6622,9:-9.9899,334:-10.2919,10544:-15.1925,6641:-15.8892,3816:-16.5653,6806:-16.936" +
        "1;13:-0.0000,21616:-13.1778,19362:-14.2946,8:-15.0012,11:-16.7598,497:-17.0044,25:-17.2109,42082:-17.5644;3816:-0.0804,3146:-2.5679,8868:-7.4468,26895" +
        ":-12.3489,2579:-13.1195,6161:-14.0899,94040:-15.1857,55908:-15.7278;198:-0.0000,320:-11.6352,11:-12.5176,345:-13.3309,720:-13.7540,374:-14.7028,482:-1" +
        "4.7816,271:-14.8803;17:-0.0000,18:-14.7252,6161:-17.8019,256:-18.1805,197:-18.3382,791:-20.0334,220:-20.0989,9673:-20.2347;13:0.0000,61259:-18.0207,88" +
        "68:-18.0931,11:-18.4636,21616:-18.5233,497:-18.5742,662:-19.2578,10544:-20.5860;8868:-0.0028,26541:-5.8702,6437:-12.7393,10544:-12.9410,38869:-14.4436" +
        ",7997:-14.5295,22725:-14.7757,56992:-14.9307;198:-0.0000,320:-13.0809,271:-15.1471,11:-15.3814,374:-16.1700,345:-17.3306,323:-17.3326,720:-17.4151;18:" +
        "-0.0000,19:-15.4497,17:-18.6080,9673:-18.7991,198:-19.3407,20:-19.5331,9:-19.8371,20215:-19.8734;13:-0.0000,26541:-12.7209,91102:-12.7689,7659:-13.314" +
        "4,48799:-15.0878,497:-16.2393,662:-16.8421,1246:-17.0445;26541:-0.0000,48799:-11.8350,816:-12.0562,14071:-13.1045,82376:-14.1939,78478:-15.7881,96179:" +
        "-16.0646,122507:-16.3056;271:-0.0002,4815:-12.4412,382:-13.9366,320:-15.2613,696:-15.6334,1432:-16.6179,198:-16.6976;7009:-0.6464,9673:-0.7795,16352:-" +
        "4.2799,791:-6.0118,10445:-7.6027,40:-8.2926,13409:-9.4902,32:-9.5925;527:-0.0033,2351:-5.7181,1051:-13.6043,636:-14.0449,2663:-15.5153,617:-16.0688,16" +
        "50:-16.3391,649:-16.7315;2663:-0.0015,6646:-6.6362,14183:-8.7342,3967:-12.0381,1650:-12.4822,61937:-12.5436,6156:-12.6085,7086:-13.8539;6156:-0.1626,3" +
        "30:-1.9007,3146:-7.9379,279:-8.9356,26150:-10.7259,721:-11.3131,66512:-11.5717,10461:-11.6157;8146:-0.6593,1606:-0.7294,369:-7.7748,1933:-9.5925,320:-" +
        "9.6902,11:-10.7090,27230:-10.7404,4245:-11.9396;1606:-0.0003,369:-8.5493,320:-9.3960,11:-9.9882,374:-12.5668,28753:-13.9073,304:-14.3542,9393:-14.5888" +
        ";814:-0.0546,1473:-3.2012,682:-4.5576,499:-7.5661,912:-8.2638,11:-8.4813,904:-8.8220,279:-8.9178;4250:-0.1536,1473:-2.6299,527:-3.2125,649:-3.5730,617" +
        ":-6.7696,3073:-8.0321,3449:-8.8083,512:-9.1958;387:0.0000,6847:-16.7139,1027:-16.7519,5694:-18.0062,3604:-18.0507,649:-18.0990,3549:-18.1555,539:-18.5" +
        "099;3549:-0.0073,1903:-5.5430,9124:-6.6982,14592:-6.7959,14454:-7.6108,9709:-8.0604,12457:-9.7597,7266:-9.9624;555:-0.0008,477:-7.5628,505:-8.4818,155" +
        "5:-10.1187,320:-11.3402,1701:-13.8061,11:-14.1959,78220:-15.1130;27890:-0.0002,35271:-8.7915,97699:-9.7081,904:-12.0130,36171:-12.2257,6651:-12.3865,2" +
        "64:-12.5326,56941:-12.8173;1023:-0.0161,1403:-4.3604,904:-6.0661,3871:-7.5749,6484:-8.2413,2204:-10.1930,477:-10.3724,9333:-10.4389;8146:-0.0001,6484:" +
        "-10.4534,9718:-10.6448,21712:-11.0750,1403:-11.5703,27230:-11.7851,24623:-12.0002,28296:-12.3103;3871:-0.0018,315:-6.3440,13:-10.2995,320:-11.9257,477" +
        ":-12.2615,11:-12.6404,32255:-14.6009,449:-15.3945;13:-0.0177,11:-4.1064,482:-7.6279,26:-8.0780,320:-8.5103,1389:-10.1815,323:-10.8377,304:-11.3846;763" +
        ":-0.3691,2435:-1.3325,12361:-3.6972,2052:-5.4613,26150:-5.7443,1472:-6.1089,1115:-6.2633,3816:-6.8700;1023:-0.0009,2144:-7.3602,28591:-9.1717,2015:-9." +
        "8240,279:-10.0552,1989:-11.1005,13168:-11.5866,2875:-11.7481;4339:-0.0000,5880:-13.0462,3492:-13.7490,24800:-15.5907,3878:-16.3591,28730:-17.1355,8146" +
        ":-17.8973,19518:-17.9370;11:-0.0000,1473:-10.4980,25:-12.6311,512:-15.6239,3638:-15.9987,26:-18.1178,13:-19.2016,814:-19.7786;814:-0.4007,499:-1.3467," +
        "6156:-3.6421,2579:-4.4807,1070:-4.8412,1521:-5.2019,422:-5.2416,912:-5.4041;527:-0.0247,4250:-4.4940,3073:-4.5296,649:-6.4227,617:-7.5810,2011:-9.1154" +
        ",2586:-9.8737,656:-10.2812;279:-0.0050,16188:-6.3163,6913:-6.5014,539:-6.9392,6646:-8.9507,330:-9.4274,264:-9.6674,659:-9.8833;6913:-0.2924,2385:-1.79" +
        "06,45648:-3.2543,16188:-3.2548,1455:-5.2706,16665:-6.9406,4857:-7.0301,330:-7.6002;4857:-0.0712,11:-3.0380,8146:-4.2781,323:-5.8341,477:-6.2740,16665:" +
        "-7.4070,6041:-7.7505,17283:-8.0403;10215:-0.0000,9612:-12.2150,2565:-12.4288,22692:-14.1715,46778:-15.2190,41840:-16.1251,50137:-17.3308,26266:-18.529" +
        "9;315:-0.0018,505:-6.6038,369:-8.5971,430:-9.0629,477:-9.1172,704:-13.2157,11:-14.3682,323:-14.4666;1933:-0.3705,682:-1.5155,8146:-3.0065,279:-3.2122," +
        "3506:-10.4192,35987:-10.6474,77678:-10.8798,904:-10.8950;11:-0.6909,13:-1.3876,323:-1.5655,430:-3.2913,505:-6.3075,10334:-8.2354,382:-8.2356,25:-8.627" +
        "9;323:-0.0088,505:-4.7675,902:-9.4056,439:-9.6019,279:-9.7772,7438:-11.2705,477:-11.2744,704:-11.6284";
    private const string LlamaCppTop8_long =
        "8586:-0.0621,334:-3.2948,16:-4.3553,29815:-5.3155,1271:-6.2919,791:-6.4590,5451:-7.5378,40:-7.5436;527:-0.0000,596:-15.0782,387:-15.9218,279:-16.7632," +
        "11503:-16.7762,1051:-16.9105,8886:-16.9972,2351:-17.4263;279:-0.0002,856:-8.7402,11503:-15.5063,2380:-15.6183,701:-18.9401,1057:-19.7223,1855:-20.2926" +
        ",1820:-21.0180;11503:-0.0001,2380:-10.0239,4860:-11.4478,4320:-12.5116,25739:-14.1395,10105:-15.4550,19089:-15.6674,3135:-17.2707;311:-0.0485,1473:-3." +
        "2077,323:-5.3714,449:-6.1626,3196:-9.5394,3235:-10.3971,11:-13.6996,512:-14.3443;279:-0.1233,1855:-2.4933,701:-3.4003,2380:-15.7738,1884:-17.9734,9739" +
        ":-18.3462,4860:-18.5336,1521:-19.2776;2380:-0.0228,4860:-3.7918,1176:-11.3359,220:-12.1536,2728:-12.9803,3488:-14.2536,11472:-14.6595,28956:-14.7215;4" +
        "860:-0.0000,3488:-14.3299,17800:-14.9097,20126:-15.6978,24271:-16.5765,67788:-17.6916,92484:-17.7063,29440:-17.8524;1473:-0.0061,3196:-5.9134,11:-5.97" +
        "15,449:-7.6788,3235:-8.1567,323:-9.6017,25393:-10.3467,382:-11.9738;16:-0.1121,334:-2.2481,5451:-7.8594,48:-10.4521,14924:-11.1735,9:-11.3149,3146:-12" +
        ".9362,791:-14.0190;13:-0.0000,19362:-13.8755,25:-15.2930,267:-15.4710,8:-15.5059,11:-17.6284,497:-17.9617,67103:-18.0774;2650:-0.7214,578:-0.7271,3781" +
        "2:-3.6211,3146:-5.9363,16225:-7.7373,3639:-8.3717,5742:-8.5396,10771:-9.5073;1690:0.0000,656:-16.8555,1790:-17.1267,9176:-17.1427,1587:-17.4132,35676:" +
        "-18.0002,1550:-18.3733,1317:-18.3822;7504:-0.0000,32249:-13.8593,3094:-16.6197,40961:-16.6329,326:-17.4005,85371:-17.5463,59846:-17.6993,1587:-18.2107" +
        ";1587:-0.0000,656:-12.5268,1550:-13.2617,19660:-14.4257,374:-14.9052,527:-14.9356,28156:-15.2146,706:-15.4185;279:-0.0000,326:-16.4969,445:-17.9820,26" +
        "4:-18.2254,435:-18.8902,578:-18.9029,3146:-19.0082,220:-19.1875;326:-0.0000,3776:-12.9134,6307:-15.1225,2579:-15.2496,445:-15.6763,6437:-16.1113,3177:" +
        "-16.1416,4251:-17.2832;59009:-0.0000,22635:-15.7127,1108:-17.7403,90644:-21.1815,326:-21.6098,56134:-21.9486,29424:-22.8989,7606:-22.9542;304:-0.0004," +
        "389:-7.9287,617:-13.2916,520:-14.5335,9932:-15.3044,315:-15.5462,449:-16.0562,7633:-16.1386;37812:-0.0000,2144:-11.3271,3146:-13.8644,17873:-14.1971,4" +
        "6083:-14.6749,38829:-14.7559,435:-15.8087,59643:-16.4089;220:-0.0000,674:-10.8193,2360:-12.9014,912:-14.2282,13:-15.9153,1396:-16.3210,4194:-16.3787,1" +
        "806:-16.8009;1806:-0.0000,1927:-14.4414,1987:-14.4434,18:-15.1908,16:-15.3009,2148:-15.4108,1114:-15.4995,220:-15.5266;617:-0.0000,320:-12.6418,706:-1" +
        "3.4014,19553:-15.2408,11:-16.2219,220:-16.2778,12522:-16.3150,304:-18.0648;1980:-0.3612,5380:-1.2280,30:-4.5851,4925:-10.2322,25:-11.5680,1473:-11.640" +
        "2,512:-12.2228,15850:-14.0168;791:-0.5145,16533:-1.0111,11439:-3.6137,22559:-5.1586,17873:-6.7898,3947:-6.9788,334:-6.9959,11286:-7.1757;326:-0.5059,4" +
        "320:-0.9256,1396:-7.9131,4495:-8.2529,3776:-10.8058,6307:-11.7453,36865:-12.0211,3488:-12.0297;59009:-0.0000,22635:-11.9171,1108:-13.1917,4970:-16.748" +
        "8,1559:-17.3031,29424:-18.0235,492:-18.3535,18400:-18.5366;304:-0.0197,389:-3.9885,9932:-7.5438,374:-8.5661,706:-8.6324,7633:-10.7322,449:-10.7752,520" +
        ":-10.8084;37812:-0.0005,2144:-8.1226,16225:-8.8351,3488:-10.7545,46083:-12.0356,38829:-12.6145,17873:-12.8317,435:-13.1814;220:-0.0000,674:-13.0929,23" +
        "60:-13.5837,912:-15.9285,1396:-15.9375,4194:-16.9888,1806:-17.0200,13:-17.0659;1806:-0.0000,1987:-11.5067,1927:-12.2885,1114:-13.3009,1544:-13.3157,21" +
        "37:-13.3576,1758:-13.5692,843:-13.8358;374:-0.1576,706:-1.9355,11:-7.4007,389:-7.5855,574:-8.8994,19813:-8.9713,320:-10.2844,17623:-10.5026;389:-0.855" +
        "2,7559:-1.0040,279:-1.6730,264:-4.0556,3776:-6.7786,7633:-7.2223,304:-7.8166,31183:-8.3449;15796:-0.0120,650:-4.7108,435:-6.3676,350:-8.3043,507:-8.64" +
        "79,279:-8.7972,29838:-8.8297,3146:-8.9407;1762:-0.0000,62479:-12.8771,375:-14.0040,1725:-14.5734,91704:-14.5793,716:-14.7013,846:-15.6172,10084:-15.65" +
        "75;88:0.0000,352:-18.9667,1065:-20.8348,85:-22.1984,379:-23.1279,68:-23.6468,12303:-23.8623,56:-23.9706;10951:-0.0000,13218:-12.0317,2209:-14.9581,230" +
        "28:-16.0681,2360:-16.4217,55365:-17.5652,320:-17.8785,9401:-17.9877;2360:-0.0757,323:-2.8398,11:-4.3901,320:-6.4665,674:-8.3453,13:-9.0076,30183:-9.63" +
        "74,449:-10.4100;13:-0.0000,220:-10.5954,2637:-12.2324,11:-14.2944,323:-15.7521,497:-15.8906,82508:-15.9701,3238:-16.0566;220:-0.0003,323:-8.6018,1806:" +
        "-9.7237,320:-12.3075,16:-13.1654,449:-13.4897,358:-13.5167,578:-13.5658;1806:-0.0002,16:-8.7544,18:-10.5160,323:-11.5888,17:-12.7735,1644:-12.8735,198" +
        "7:-12.8809,1927:-12.9631;323:-0.5733,11:-1.0463,13:-2.4837,320:-7.3044,706:-7.5378,449:-8.1267,374:-9.3490,382:-10.0928;706:-0.4407,433:-1.7245,374:-1" +
        ".8696,574:-4.1526,279:-6.2606,37812:-6.2660,4184:-6.3513,1202:-6.6670;220:-0.0027,264:-6.7780,3146:-7.5968,3776:-7.6616,1027:-8.8941,279:-9.0238,330:-" +
        "9.0501,459:-10.3633;11286:-0.0001,10513:-11.1106,10895:-11.2913,3776:-12.1561,1806:-12.6200,9892:-12.8331,2970:-12.9824,1987:-13.2043;7504:-0.0001,322" +
        "49:-9.7787,3776:-13.6876,40961:-14.6233,25047:-14.9230,3094:-15.2333,85371:-15.6022,13:-16.1895;382:-0.9561,13:-1.0679,11:-1.7277,323:-2.5786,320:-4.1" +
        "412,4184:-6.3907,574:-8.3993,439:-8.7553;17:-0.2723,40:-2.1394,1271:-3.6957,6788:-3.9382,70869:-4.0398,16533:-4.1338,4438:-4.2545,52522:-4.5784;13:-0." +
        "0000,5450:-11.7466,11:-14.2216,303:-14.3030,763:-14.7959,497:-15.0121,662:-15.1891,64:-15.2148;763:-0.0000,3639:-10.6086,3277:-12.5020,304:-12.7580,22" +
        "0:-13.2348,320:-13.3121,2650:-13.3784,578:-15.2065;1148:-0.0000,902:-15.1590,3639:-16.8923,12840:-18.3179,279:-19.9949,2144:-20.0068,35389:-20.9726,38" +
        "535:-21.1693;1060:0.0000,9941:-17.3423,574:-18.7122,1667:-18.7841,25418:-19.2630,3236:-19.3619,14987:-20.3962,20043:-20.8392;574:-0.0000,1550:-15.1388" +
        ",1051:-15.6791,374:-16.1506,320:-18.1390,1587:-18.1617,279:-18.3984,15148:-18.5441;279:-0.0000,326:-15.9783,5918:-16.1864,264:-16.7787,445:-17.0154,42" +
        "0:-17.2555,37812:-17.4523,220:-17.6085;326:-0.0000,3177:-14.4897,445:-14.8059,220:-15.9330,95911:-16.0156,75:-16.4959,36865:-17.0161,55372:-17.0777;59" +
        "009:0.0000,22635:-16.8054,1108:-18.4311,56134:-22.2933,7606:-22.9605,90644:-23.0182,29424:-23.2798,20870:-23.3283;304:-0.0000,389:-11.8647,9932:-16.12" +
        "65,320:-16.1311,315:-16.3372,520:-16.7455,220:-17.1490,7633:-17.4325;37812:-0.0000,2144:-13.5235,17873:-14.8898,38829:-15.7184,46083:-15.8601,220:-17." +
        "2081,435:-17.7896,59643:-18.3478;220:-0.0000,5538:-14.6299,2360:-15.3524,320:-15.9298,912:-16.4848,13:-17.1988,15235:-17.2183,674:-17.4542;5538:-0.000" +
        "5,4103:-9.1202,1313:-9.5112,2983:-9.5227,717:-9.8710,843:-10.1046,17:-11.4690,21:-11.6814;5918:-0.0000,1977:-12.0997,320:-12.6007,220:-14.4319,60685:-" +
        "14.6729,574:-14.8275,52714:-15.0419,11:-15.1399;1980:-0.0000,30:-10.2399,12241:-13.1787,58877:-13.2788,11:-14.5992,323:-14.6666,69113:-14.8520,27948:-" +
        "14.8678;791:-0.0209,11439:-4.3723,17873:-4.9275,40:-8.5183,3915:-9.1153,2028:-9.5237,3947:-9.5943,644:-10.1441;326:-0.0009,1060:-7.3073,4320:-9.0194,2" +
        "144:-10.8590,2038:-11.3973,68690:-11.7215,220:-12.0562,1396:-12.1556";

    private static (int Id, float LogProb)[][] ParseTop8(string s) => s.Split(';').Select(step => step.Split(',').Select(c =>
    {
        var kv = c.Split(':');
        return (int.Parse(kv[0]), float.Parse(kv[1], System.Globalization.CultureInfo.InvariantCulture));
    }).ToArray()).ToArray();

    private static string LighthousePrompt()
    {
        string[] colors = { "red", "white", "black", "striped", "grey", "green", "blue", "yellow" };
        string[] islands = { "Kelp", "Fennel", "Osprey", "Marrow", "Tansy", "Quill", "Brine", "Larch", "Sorrel", "Dunlin", "Hazel", "Vetch", "Rook", "Sedge", "Wren" };
        var sb = new StringBuilder("Here is a list of facts.\n");
        for (int i = 0; i < LighthouseFacts.Length / 9; i++)
        {
            var f = LighthouseFacts.AsSpan(i * 9, 9);
            if (i > 0) sb.Append('\n');
            sb.Append($"Fact {i + 1}: The {colors[f[0] - '0']} lighthouse on {islands[Convert.ToInt32(f[1].ToString(), 16)]} Island No. {i + 1} has {int.Parse(f.Slice(2, 3))} steps and was built in {int.Parse(f.Slice(5, 4))}.");
        }
        sb.Append("\n\nUsing only the facts above, answer three questions. First, how many steps does the lighthouse in Fact 37 have? " +
                  "Second, in what year was the lighthouse in Fact 62 built? Third, what colour is the lighthouse in Fact 9? Then explain briefly how you found each answer.");
        return sb.ToString();
    }

    /// <summary>
    /// Llama 3.x carries per-frequency rotary factors (rope_freqs.weight) that divide each pair's angle (ggml: theta / freq_factors[i], on every llama layer).
    /// The session engine applied them only to gemma4's global layers and the batch decoder refused the model (Data, SpawnDev.AI, 2026-11-02; Vertex runs
    /// llama3.1:8b). Reference: Ollama 0.34.4 (llama.cpp) on this machine's llama3.1:8b, raw prompt (the Llama 3 template text with no system turn; Ollama adds
    /// the BOS itself, so the text sent leaves out &lt;|begin_of_text|&gt;), temperature 0, top_k 1, repeat_penalty 1, num_ctx 8192. A short prompt (22 tokens) and a
    /// long one (2194 tokens: 75 facts and three questions; past position ~2048 the factors change the angle of the slow pairs by up to 8x). The token counts
    /// equal Meta's tiktoken tokenizer rebuilt from the GGUF vocabulary (a first reference with num_ctx 4096 had been truncated by Ollama without a word). Each engine teacher-forced along llama.cpp's answer must give llama.cpp's top-8 candidates at every step the log-probabilities llama.cpp gives them
    /// (a free-running greedy answer forks at the first near-tie), the short case's greedy text must equal llama.cpp's, and the batch decoder must load the model.
    /// Desktop lanes (the browser cannot read the Ollama cache); skipped when llama3.1:8b is not cached.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Llama3RopeFreqs_BothEngines_MatchLlamaCpp() => await RunTest(async accelerator =>
    {
        if (OperatingSystem.IsBrowser() || accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("reads the local Ollama cache (CUDA / OpenCL lanes)");
        string root = Environment.GetEnvironmentVariable("OLLAMA_MODELS") is { Length: > 0 } env ? env
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".ollama", "models");
        var manifest = Path.Combine(root, "manifests", "registry.ollama.ai", "library", "llama3.1", "8b");
        if (!File.Exists(manifest)) throw new UnsupportedTestException("llama3.1:8b not in the Ollama cache");
        string? path = null;
        using (var doc = JsonDocument.Parse(File.ReadAllText(manifest)))
            foreach (var layer in doc.RootElement.GetProperty("layers").EnumerateArray())
                if (layer.GetProperty("mediaType").GetString()!.EndsWith(".model")) path = Path.Combine(root, "blobs", layer.GetProperty("digest").GetString()!.Replace(':', '-'));
        if (path == null || !File.Exists(path)) throw new UnsupportedTestException("llama3.1:8b blob missing");

        var cases = new (string Name, string Content, int PromptTokens, int MaxNew, string Want)[]
        {
            ("short", "Name three primary colors and say why they are called primary.", 22, 48,
                "The three primary colors are:\n\n1. Red\n2. Blue\n3. Yellow\n\nThey are called primary colors because they cannot be created by mixing other colors together. In other words, they are the basic building blocks of color, and"),
            ("long", LighthousePrompt(), 2194, 64,
                "Here are the answers to the three questions:\n\n1. How many steps does the lighthouse in Fact 37 have?\n\nThe lighthouse in Fact 37 is on Osprey Island No. 37 and has 158 steps.\n\n2. In what year was the lighthouse in Fact 62 built?\n\nThe l"),
        };
        const int MaxSeq = 2304;
        // llama.cpp's greedy pick may differ from ours where two tokens nearly tie (long step 11: " How" -0.7214 against " The" -0.7271 in llama.cpp itself),
        // and one different pick changes everything after it; a top-1 check also misses the factors (MEASURED: without rope_freqs the 2194-token answer
        // keeps 63 of 64 picks). So each engine is teacher-forced along llama.cpp's answer and its log-softmax compared with llama.cpp's log-probabilities
        // for llama.cpp's top-8 candidates at every step (those above LpFloor: far tails carry no mass and amplify noise). Free-running greedy must also
        // equal llama.cpp's text where no tie occurs (the short case).
        // MEASURED 2026-11-02 CUDA, long case: mean 0.19 / max 1.25 with the factors, 0.47 / 4.08 without them (either engine); short: 0.06 / 0.33
        const float LpFloor = -10f, TolMean = 0.3f, TolMax = 2f;
        var prompts = new int[cases.Length][];
        var refIds = new int[cases.Length][];
        var gaps = new Dictionary<string, float[]>();
        var lpErr = new Dictionary<string, List<float>>();
        var top8 = new[] { ParseTop8(LlamaCppTop8_short), ParseTop8(LlamaCppTop8_long) };
        static float[] LogSoftmax(float[] row)
        {
            float m = row.Max(); double sum = 0; foreach (var v in row) sum += Math.Exp(v - m);
            float lse = m + (float)Math.Log(sum);
            return row.Select(v => v - lse).ToArray();
        }
        void Compare(string key, int c, int i, float[] row)
        {
            if (!lpErr.TryGetValue(key, out var l)) lpErr[key] = l = new List<float>();
            var ls = LogSoftmax(row);
            foreach (var (id, lp) in top8[c][i]) if (lp >= LpFloor) l.Add(Math.Abs(ls[id] - lp));
        }
        var greedy = new Dictionary<string, string>();
        HashSet<int> stops;
        SentencePieceTokenizer tok;
        GgufEngine autoEngine; string autoReason;
        static (int Best, float Gap) Score(float[] row, int want)
        {
            int b = 0; for (int i = 1; i < row.Length; i++) if (row[i] > row[b]) b = i;
            return (b, row[b] - row[want]);
        }
        using (var gen = await GgufTextGenerationPipeline.CreateFromFileAsync(accelerator, path, maxSeqLen: MaxSeq, engine: GgufEngine.Generator))
        {
            var session = gen.Session; tok = gen.Tokenizer;
            stops = new HashSet<int> { tok.EosId };
            foreach (var s in new[] { "<|eot_id|>", "<|end_of_text|>", "<|eom_id|>" }) if (tok.TryGetId(s, out var id)) stops.Add(id);
            using var inBuf = accelerator.Allocate1D<float>(MaxSeq);
            async Task<float[]> Step(int[] tokens)
            {
                inBuf.View.SubView(0, tokens.Length).CopyFromCPU(tokens.Select(t => (float)t).ToArray());
                var o = await session.RunDecodeStepAsync(new Dictionary<string, Tensor> { ["input_ids"] = new Tensor(inBuf.View.SubView(0, tokens.Length), new[] { 1, tokens.Length }, "input_ids") });
                var t = o.TryGetValue("logits", out var l) ? l : o.Values.First();
                int vocab = t.Shape[^1], rows = t.ElementCount / vocab;
                return await t.Data.SubView((long)(rows - 1) * vocab, vocab).CopyToHostAsync();
            }
            for (int c = 0; c < cases.Length; c++)
            {
                prompts[c] = ChatTemplates.BuildLlama3PromptTokens(tok, new List<(string, string)> { ("user", cases[c].Content) });
                if (prompts[c].Length != cases[c].PromptTokens)
                    throw new Exception($"{cases[c].Name}: our prompt is {prompts[c].Length} tokens, llama.cpp read {cases[c].PromptTokens} (the rebuilt text or the tokenizer differs)");
                refIds[c] = tok.Encode(cases[c].Want);
                if (refIds[c].Length != cases[c].MaxNew || tok.Decode(refIds[c]) != cases[c].Want)
                    throw new Exception($"{cases[c].Name}: llama.cpp's answer encodes to {refIds[c].Length} tokens here (it generated {cases[c].MaxNew}) or does not decode back");
                if (top8[c].Length != refIds[c].Length || Enumerable.Range(0, refIds[c].Length).Any(i => top8[c][i][0].Id != refIds[c][i]))
                    throw new Exception($"{cases[c].Name}: llama.cpp's chosen tokens (the top-1 of each step) are not our encoding of its answer");
                // teacher-forced along llama.cpp's answer
                session.ResetGGUFDecode();
                var g = new float[refIds[c].Length];
                var row = await Step(prompts[c]);
                for (int i = 0; i < refIds[c].Length; i++)
                {
                    g[i] = Score(row, refIds[c][i]).Gap;
                    Compare($"{cases[c].Name} session", c, i, row);
                    if (i + 1 < refIds[c].Length) row = await Step(new[] { refIds[c][i] });
                }
                gaps[$"{cases[c].Name} session"] = g;
                // free-running greedy
                session.ResetGGUFDecode();
                var ids = new List<int>();
                row = await Step(prompts[c]);
                while (true)
                {
                    int next = Score(row, 0).Best;
                    if (stops.Contains(next)) break;
                    ids.Add(next);
                    if (ids.Count == cases[c].MaxNew) break;
                    row = await Step(new[] { next });
                }
                greedy[$"{cases[c].Name} session"] = tok.Decode(ids.ToArray());
            }
        }
        using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true))
        using (var dec = await GgufBatchDecoder.CreateAsync(accelerator, fs, maxSeq: MaxSeq, maxBatch: 1))
            for (int c = 0; c < cases.Length; c++)
            {
                var g = new float[refIds[c].Length];
                await dec.PrefillAsync(prompts[c]);
                var row = await dec.ReadLogitsAsync();
                for (int i = 0; i < refIds[c].Length; i++)
                {
                    g[i] = Score(row, refIds[c][i]).Gap;
                    Compare($"{cases[c].Name} batch", c, i, row);
                    if (i + 1 < refIds[c].Length) { await dec.StepAsync(new[] { refIds[c][i] }); row = await dec.ReadLogitsAsync(); }
                }
                gaps[$"{cases[c].Name} batch"] = g;
                greedy[$"{cases[c].Name} batch"] = tok.Decode((await dec.GenerateAsync(prompts[c], 1, cases[c].MaxNew, 0f, 1f, 1, stops))[0]);
            }
        using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true))
        {
            var model = await SpawnDev.ILGPU.ML.GGUF.GGUFParser.ParseHeaderAsync(fs);
            autoEngine = GgufBatchDecoder.Supports(model, out autoReason) ? GgufEngine.BatchDecoder : GgufEngine.Generator;
        }

        static int Common(string a, string b) { int i = 0; while (i < a.Length && i < b.Length && a[i] == b[i]) i++; return i; }
        string Show(string s) => s.Replace("\n", "\\n");
        bool ok = autoEngine == GgufEngine.BatchDecoder;
        var notes = new List<string> { $"batch decoder loads llama3.1:8b: {(ok ? "yes" : "NO (" + autoReason + ")")}" };
        for (int c = 0; c < cases.Length; c++)
            foreach (var engine in new[] { "session", "batch" })
            {
                string key = $"{cases[c].Name} {engine}";
                var g = gaps[key];
                int worst = 0; for (int i = 1; i < g.Length; i++) if (g[i] > g[worst]) worst = i;
                int off = g.Count(x => x > 0);
                var e = lpErr[key];
                float mean = e.Average(), max = e.Max();
                bool forcedOk = mean <= TolMean && max <= TolMax;
                ok &= forcedOk;
                string text = greedy[key], want = cases[c].Want;
                bool same = text == want;
                if (cases[c].Name == "short") ok &= same;
                int k = Common(text, want);
                notes.Add($"{key}: log-probabilities vs llama.cpp's mean |diff| {mean:F4} max {max:F3} over {e.Count} ({(forcedOk ? "within" : "OUTSIDE")} {TolMean} / {TolMax}); " +
                    $"{off} of {g.Length} of llama.cpp's picks not our top-1 (worst by {g[worst]:F3} at token {worst}); greedy " +
                    (same ? "= llama.cpp" : $"differs at char {k}: ours ...{Show(text[Math.Max(0, k - 20)..Math.Min(text.Length, k + 30)])} | llama.cpp ...{Show(want[Math.Max(0, k - 20)..Math.Min(want.Length, k + 30)])}"));
            }
        var report = $"{accelerator.AcceleratorType}: " + string.Join(" | ", notes);
        Console.WriteLine("[Llama3RopeFreqs] " + report);
        if (!ok) throw new Exception(report);
        return report;
    });
}
