#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <cmath> 


#include "TCut.h"
#include "TBranch.h"
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TLegend.h"
#include "TArrow.h"  
#include "TLatex.h"  
#include <TRandom3.h>
#include "TGraph.h"  

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

#include "RooRealVar.h"
#include "RooConstVar.h"
#include "RooGaussian.h"
#include "RooAddPdf.h"
#include <RooArgusBG.h>
#include "RooDataSet.h"
#include "RooPlot.h"
#include "RooFitResult.h"  
#include "RooAbsPdf.h"     
#include "RooExponential.h" 

using namespace RooFit;


TString generateRandomString(int length) {
    const std::string characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    TString randomString;
    TRandom3 randGen(0);
    
    for (int i = 0; i < length; ++i) {
        int index = randGen.Integer(characters.size());
        randomString += characters[index];
    }
    
    return randomString;
}



/**
 * @brief Does the Fit and Visualizes.
 *
 * @param chain        TChain of events.
 * @param BDT_cut_val  progressive BDT cut.
 * @param other_cuts   any additional selection.
 * @param fit_range    [min, max] fit range.
 * @param nBins        histogram bin count.
 * @param figure_name  output image name.
 * @return             Pointer to array: {Nsig, Nbkg}.
 */



Double_t* fit(TChain* chain, Double_t BDT_cut_val, TCut other_cuts, 
              Double_t fit_range[2], int nBins, TString figure_name, 
              TString Fit_version, TString Info_text)
{
    std::cout << "Pikachu Number 3" << std::endl;

    // Define the BDT cut
    TCut BDT_cut = TCut(TString::Format("Res_BDT > %.2f", BDT_cut_val));

    // Generate temporary file for RooFit operations
    TString randomString = generateRandomString(10);
    TString current_Dir = "/home/alexanum/WORKSPACE/testing/ANALYSIS/Output/B_BDT/FoMScan/";
    TString tempFilePath = current_Dir + "temp_" + randomString + ".root";
    TFile* tempFile = new TFile(tempFilePath, "recreate");
    tempFile->cd();

    // Copy filtered TTree
    TTree* tempTree = chain->CopyTree(BDT_cut + other_cuts);
    std::cout << "Entries after cut: " << tempTree->GetEntries() << std::endl;

    // Define all variables used in RooFit
    RooRealVar B_M("B_M", "Invariant Mass Spectrum of B", fit_range[0], fit_range[1]);
    RooRealVar Res_BDT("Res_BDT", "BDT response", -1, 1);
    RooRealVar D0_M("D0_M", "D0 mass", 1700, 2000); // adjust range if needed

    // Create RooDataSet including all relevant branches
    RooDataSet data("data", "B_M dataset", tempTree, RooArgSet(B_M, Res_BDT, D0_M));

    // Define fitting parameters
    RooRealVar mean("mean", "mean of gaussian", 5279, 5270, 5280);
    RooRealVar sigma("sigma", "width of gaussian", 10, 0.1, 25);
    RooRealVar decay("decay", "decay constant", -0.0001, -100, 0.0);
    RooRealVar argpar("argpar", "argus shape parameter", -50, -200, -1.);
    RooRealVar argcutoff("argcutoff", "argus cutoff", 6500.0);

    // Build PDFs
    RooAbsPdf* sig = new RooGaussian("sig", "Gaussian PDF", B_M, mean, sigma);
    RooAbsPdf* bkg = new RooExponential("bkg", "Exponential PDF", B_M, decay);

    RooRealVar nsig("nsig", "number of signal events", 1000, 0., 10000);
    RooRealVar nbkg("nbkg", "number of background events", 1000, 0, 50000);

    RooAddPdf model("model", "sig+bkg", RooArgList(*bkg, *sig), RooArgList(nbkg, nsig));

    // Perform the fit
    RooFitResult* result = model.fitTo(data, RooFit::PrintLevel(-1), RooFit::Warnings(false));

    // Integration in signal region
    Double_t SRwidth = 40.0; // half the SR width
    B_M.setRange("signal", mean.getVal() - SRwidth, mean.getVal() + SRwidth);
    RooAbsReal* Nsig = sig->createIntegral(B_M, NormSet(B_M), Range("signal"));
    RooAbsReal* Nbkg = bkg->createIntegral(B_M, NormSet(B_M), Range("signal"));

    std::cout << "Nsig: " << nsig.getVal() * Nsig->getVal() << std::endl;
    std::cout << "Nbkg: " << nbkg.getVal() * Nbkg->getVal() << std::endl;

    // Visualization
    TCanvas* c1 = new TCanvas("c1", "Fit Canvas", 800, 600);
    RooPlot* xframe = B_M.frame();
    data.plotOn(xframe, Binning(nBins), Name("Plot_data"), MarkerColor(kBlack));
    model.plotOn(xframe, Name("Plot_model"), LineColor(kRed), LineWidth(2));
    model.plotOn(xframe, Components(RooArgSet(*sig)), LineColor(kGreen), LineStyle(kDashed), Name("Plot_signal"));
    model.plotOn(xframe, Components(RooArgSet(*bkg)), LineColor(kBlue), LineStyle(kDashed), Name("Plot_background"));
    xframe->SetTitle("B Mass " + Fit_version + " Fit");
    xframe->GetXaxis()->SetTitle("m_{B} [MeV/c^{2}]");
    xframe->GetYaxis()->SetTitle("Entries / bin");
    xframe->Draw();

    TLegend* legend = new TLegend(0.65, 0.7, 0.88, 0.88);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->AddEntry(xframe->findObject("Plot_data"), "Data", "lep");
    legend->AddEntry(xframe->findObject("Plot_model"), "SIG + BKG", "l");
    legend->AddEntry(xframe->findObject("Plot_signal"), "SIG (GAUSS)", "l");
    legend->AddEntry(xframe->findObject("Plot_background"), "BKG (EXP)", "l");
    legend->Draw();

    TLatex z;
    z.SetNDC(kTRUE);
    z.SetTextSize(0.04);
    z.DrawLatex(0.62, 0.5, Info_text);

    c1->SaveAs(figure_name);

    // Prepare results
    Double_t* integral_result = new Double_t[2];
    integral_result[0] = nsig.getVal() * Nsig->getVal();
    integral_result[1] = nbkg.getVal() * Nbkg->getVal();

    // Cleanup
    delete result;
    delete Nsig;
    delete Nbkg;
    delete sig;
    delete bkg;
    delete xframe;
    delete legend;
    delete c1;
    tempFile->Close();
    delete tempFile;
    // std::remove(tempFilePath.Data());

    return integral_result;
}


int FoMScan(TString datafilepaths, TString MCfilepaths, TString fig_dir, TString outfilepath){

    TString channel;
    TString track;
    TString version;
    TString FoMScanLatex;
    TCut cut_D0_loose;
    TCut cut_B_M_sb;

    int nBins = 50;
    version = "scan";
    if (datafilepaths.Contains("D2H")){
        channel = "D2H";
        cut_D0_loose = "abs(D0_M-1864.84) < 50";
        if (datafilepaths.Contains("DD")){
            track = "DD";
            cut_B_M_sb = "B_M > 5500 && B_M < 7000"; // In case we need to be able to extend the SB for LL
            FoMScanLatex = "#splitline{#font[10]{LHCb Run 2, Track: DD}}{#font[10]{B^{-}#rightarrow#bar{p}#Lambda(D^{0}#rightarrowK#pi)}}";

        }
        else if (datafilepaths.Contains("LL")){
            track = "LL";
            cut_B_M_sb = "B_M > 5500 && B_M < 7000"; // In case we need to be able to extend the SB for LL
            FoMScanLatex = "#splitline{#font[10]{LHCb Run 2, Track: LL}}{#font[10]{B^{-}#rightarrow#bar{p}#Lambda(D^{0}#rightarrowK#pi)}}";
        }
        
    }

    else if (datafilepaths.Contains("D4H")){
        channel = "D4H";
        cut_D0_loose = "abs(D0_M-1869.5) < 50";
        if (datafilepaths.Contains("DD")){
            track = "DD";
            cut_B_M_sb = "B_M > 5500 && B_M < 7000";
            //FoMScanLatex = "#splitline{#font[10]{LHCb Run 2, Track: DD}}{#font[10]{B^{-}#rightarrow #Lambda #bar{p} (D^{0}#rightarrowK#pi#pi#pi) }}";
            FoMScanLatex = "#splitline{#font[10]{LHCb Run 2, Track: DD}}{#font[10]{B^{-}#rightarrow#bar{p}#Lambda(D^{0}#rightarrowK#pi#pi#pi) }}";
        }
        else if (datafilepaths.Contains("LL")){
            track = "LL";
            cut_B_M_sb = "B_M > 5500 && B_M < 7000";
            FoMScanLatex = "#splitline{#font[10]{LHCb Run 2, Track: LL}}{#font[10]{B^{-}#rightarrow#bar{p}#Lambda(D^{0}#rightarrowK#pi#pi#pi) }}";
        }
        
    }

    std::cout<<"Started FOM scan"<<std::endl;

    TCut cut_sig = cut_D0_loose;
    TCut cut_bkg_sb = cut_D0_loose + cut_B_M_sb;

    // Initiate with a very small BDT cut
    Double_t BTD_cutVal = -0.6;
    TCut cut_BDT_loose = Form("Res_BDT > %f", BTD_cutVal);
                                      // old value: 5100 changed to 5200, min the lower SB 
    Double_t data_sig_fit_range[2] = {5200, 6000};  // ragne for fitting model (not B0)
    TString fig_fit_raw = fig_dir + "raw_fit_" + channel + "_" + track + "_" + version + ".png";
    TString fig_fit_opt = fig_dir + "optimised_fit_" + channel + "_" + track + "_" + version + ".png";

    //-------------------------------
    // FOM calculation
    //-------------------------------
    // Data
    // ﹂Polluted SR ∈ signal S0 and bkg component B0, Range: NominalBmass (5300MeV) ± 40MeV (hardcoded in Fit/Integration section)
    // ﹂Extended SB ∈ bkg B0', Range: cut_B_M_sb
    // MC Simu
    // ﹂SR ∈ S0', Range: NominalBmass (5300MeV) ± 40MeV (hardcoded in Fit/Integration section)
    // fs = S0 / S0' setconst
    // fb = B0 / B0' setConst
    // Iterate through BDTcuts
    //  > get new S0', B0'
    //  > S = S0' * fs
    //  > B = B0' * fb
    //  > FOM = S / sqrt(S + B) or S / sqrt(S + B) * S / (S + B)

    TChain* dataChain = new TChain("DecayTree");
    dataChain->Add(datafilepaths);

    //Debugging:
    std::cout << "FILEPATHS" << std::endl;
    std::cout << datafilepaths << std::endl;
    std::cout << MCfilepaths << std::endl;

    Double_t Nbkg_sb = dataChain->GetEntries(cut_BDT_loose + cut_bkg_sb); // ➤➤➤ B0'

    TChain* MCChain = new TChain("DecayTree");
    MCChain->Add(MCfilepaths);
    // DO NOT APPLY: this is the bastard that fked up the FoM curve: TCut MC_cut = "B_M > 5250 && B_M < 5290";
    Double_t Nsig_MC = MCChain->GetEntries(cut_BDT_loose + cut_sig); // ➤➤➤ S0'

    std::cout << "Nsig in MC: " << Nsig_MC << std::endl;
    std::cout << "Nbkg in data sideband: " << Nbkg_sb << std::endl;

    // fit parameters:                   TChain,  BDTcut, other cuts, fitting range, nBins, figure name
    Double_t* integral_result_raw = fit(dataChain, BTD_cutVal, cut_sig, data_sig_fit_range, nBins, fig_fit_raw, "Raw", FoMScanLatex);
    Double_t Nsig = integral_result_raw[0]; // ➤➤➤ S0
    Double_t Nbkg = integral_result_raw[1]; // ➤➤➤ B0
    Double_t sig_scale = Nsig/Nsig_MC;      // fs
    Double_t bkg_scale = Nbkg/Nbkg_sb;      // fb
    std::cout << "Scale of signal: " << sig_scale << std::endl;
    std::cout << "Scale of background: " << bkg_scale << std::endl;

    // Progeress through different BDTcuts (Res_BDT > ...) and determine the most optimal 

    Double_t BDT_opt = -1;
    Double_t FoM_max = 0;
    Double_t FoM;
    Double_t BDT_array[60];
    Double_t FoM_array[60];

    int count = 0;
    for (Double_t BDT = BTD_cutVal; BDT <= 0.2; BDT = BDT + 0.01){
        if (count >= 80) break;
        if (!std::isfinite(BDT)) continue;
        TCut cut_BDT_scan = TCut(TString::Format("(Res_BDT > %f)", BDT));
	    Double_t Nsig_MC_scan = MCChain->GetEntries(cut_BDT_scan + cut_sig);      // ➤➤➤ S0'
	    Double_t Nbkg_sb_scan = dataChain->GetEntries(cut_BDT_scan + cut_bkg_sb); // ➤➤➤ B0'
        Double_t S = Nsig_MC_scan * sig_scale;                                   
        Double_t B = Nbkg_sb_scan * bkg_scale;                                   
        // Double_t FoM = S/sqrt(S+B) * ;
        Double_t FoM = S/sqrt(S+B) * (S/(S+B));

        BDT_array[count] = BDT;
        FoM_array[count] = FoM;

        count = count + 1;
        std::cout << "BDT= " << BDT << ", S= " << S << ", B= " << B << ", FoM= " << FoM << std::endl;
        if (FoM > FoM_max){
            FoM_max = FoM;
            BDT_opt = BDT;
        }
    }

    gSystem->Exec("rm -f /home/alexanum/WORKSPACE/testing/ANALYSIS/Output/B_BDT/FoMScan/temp_*.root");
    std::cout << "Maximum FoM: " << FoM_max << std::endl;
    std::cout << "Optimized BDT: " << BDT_opt << std::endl;
    
    // Apply the most Optimal BDTcut
    // fit parameters:                    TChain,   BDTcut,  other cuts, fitting range,   nBins, figure name
    Double_t* integral_result_opt = fit(dataChain, BDT_opt, cut_sig, data_sig_fit_range, nBins, fig_fit_opt, "Optimized", FoMScanLatex);


    // Visualize the FoM and BDTcut progress
    TCanvas *c2 = new TCanvas();
    TGraph* g = new TGraph(60, BDT_array, FoM_array);
    g->SetTitle("FoM Scan: Progressive BDT Cut; BDT Cut; S/#sqrt{S+B} #times S/(S+B)");
    g->Draw();
    auto a = new TArrow(BDT_opt, FoM_max*0.8, BDT_opt, FoM_max);
    a->SetArrowSize(0.02);
    a->SetLineColor(2);
    a->SetFillColor(2);
    a->Draw();
    TLatex t;
    t.SetNDC(kTRUE);
    t.SetTextSize(0.04);
    
    t.DrawLatex(0.15, 0.5, FoMScanLatex);   //before 0.15
    c2->SaveAs(fig_dir + "Scan_" + channel + "_" + track + "_" + version +".png");

    cout<<"BDT: "<<BDT_opt<<endl;
    cout<<"FoM: "<<FoM_max<<endl;

	std::ofstream outfile(outfilepath);
	if(outfile.is_open()){
		outfile<<BDT_opt<< std::endl;
	}

    outfile.close();
    delete c2;
    delete dataChain;
    delete MCChain;
    delete[] integral_result_raw;
    delete[] integral_result_opt;   
    return 0;
} 
