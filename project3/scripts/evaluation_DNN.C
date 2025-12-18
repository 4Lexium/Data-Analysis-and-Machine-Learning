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


Double_t* fit(TChain* chain, Double_t BDT_cut_val, TCut other_cuts, 
              Double_t fit_range[2], int nBins, TString figure_name, 
              TString Fit_version, TString Info_text)
{
    std::cout << "Pikachu Number 3" << std::endl;

    // Define the BDT cut
    TCut BDT_cut = TCut(TString::Format("Res_BDT > %.2f", BDT_cut_val));

    // Generate temporary file for RooFit operations
    TString randomString = generateRandomString(10);
    TString current_Dir = "/home/alexanum/WORKSPACE/testing/ANALYSIS/Output/B_DNN/FoMScan/";
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

    // Prepare results
    Double_t* integral_result = new Double_t[2];
    integral_result[0] = nsig.getVal() * Nsig->getVal();
    integral_result[1] = nbkg.getVal() * Nbkg->getVal();

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
    std::cout<< "Figure Saved" <<std::endl;

    // Cleanup
    delete result;
    delete Nsig;
    delete Nbkg;
    delete sig;
    delete bkg;

    c1->Clear();
    c1->Close();
    delete c1;

    c1->Clear();        // remove all primitives including xframe
    gPad->Modified();   // force refresh
    gPad->Update();
    std::cout<<"pikaaaa"<<std::endl;
    delete xframe;
    delete legend;
    delete c1;

    tempFile->Close();
    delete tempFile;
    // std::remove(tempFilePath.Data());

    return integral_result;
}

// =====================================================================
//              RUN A SINGLE FIXED BDT CUT, NO SCAN
// =====================================================================
int FoMSingleCut()
{
    //------------------------------------------------------------
    // User settings
    //------------------------------------------------------------
    TString datafilepaths = "/home/alexanum/WORKSPACE/testing/ANALYSIS/Output/B_BDT/TMVAApplication/data_*_D4H_*_DD.root";
    TString MCfilepaths   = "/home/alexanum/WORKSPACE/testing/ANALYSIS/Output/B_BDT/TMVAApplication/mc_*_D4H_*_DD.root";
    TString fig_dir       = "/home/alexanum/WORKSPACE/testing/ANALYSIS/Output/Evaluation/FoMScan/SingleCut/";
    TString outfile       = "/home/alexanum/WORKSPACE/testing/ANALYSIS/Output/Evaluation/FoMScan/SingleCut/SingleCut_D4H_DD.txt";  //has to be left like this its Roo magic

    gSystem->Exec(Form("mkdir -p %s", fig_dir.Data()));

    //------------------------------------------------------------
    // Fixed BDT cut
    //------------------------------------------------------------
    double BDTcut = 0.8;   // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    TCut cut_BDT = Form("Res_BDT > %f", BDTcut);

    //------------------------------------------------------------
    // Physics selection cuts
    //------------------------------------------------------------
    TCut cut_D0     = "abs(D0_M - 1864.84) < 50";
    TCut cut_B_sb   = "B_M > 5500 && B_M < 7000";  // sideband (background)

    //------------------------------------------------------------
    // Chains
    //------------------------------------------------------------
    TChain* dataChain = new TChain("DecayTree");
    dataChain->Add(datafilepaths);

    TChain* MCChain = new TChain("DecayTree");
    MCChain->Add(MCfilepaths);

    //------------------------------------------------------------
    // Fit parameters
    //------------------------------------------------------------
    double fit_range[2] = {5200, 6000};
    int nBins = 50;

    TString fig_name = fig_dir + "Fit_D4H_DD_SingleCut.png";

    TString latexInfo;

    if (datafilepaths.Contains("D2H")){
        // channel = "D2H";
        // cut_D0_loose = "abs(D0_M-1864.84) < 50";
        if (datafilepaths.Contains("DD")){
            // track = "DD";
            // cut_B_M_sb = "B_M > 5500 && B_M < 7000"; // In case we need to be able to extend the SB for LL
            latexInfo = "#splitline{#font[10]{LHCb Run 2, Track: DD}}{#font[10]{B^{-}#rightarrow#bar{p}#Lambda(D^{0}#rightarrowK#pi)}}";

        }
        else if (datafilepaths.Contains("LL")){
            // track = "LL";
            // cut_B_M_sb = "B_M > 5500 && B_M < 7000"; // In case we need to be able to extend the SB for LL
            latexInfo = "#splitline{#font[10]{LHCb Run 2, Track: LL}}{#font[10]{B^{-}#rightarrow#bar{p}#Lambda(D^{0}#rightarrowK#pi)}}";
        }
        
    }

    else if (datafilepaths.Contains("D4H")){
        // channel = "D4H";
        // cut_D0_loose = "abs(D0_M-1869.5) < 50";
        if (datafilepaths.Contains("DD")){
            // track = "DD";
            // cut_B_M_sb = "B_M > 5500 && B_M < 7000";
            //FoMScanLatex = "#splitline{#font[10]{LHCb Run 2, Track: DD}}{#font[10]{B^{-}#rightarrow #Lambda #bar{p} (D^{0}#rightarrowK#pi#pi#pi) }}";
            latexInfo = "#splitline{#font[10]{LHCb Run 2, Track: DD}}{#font[10]{B^{-}#rightarrow#bar{p}#Lambda(D^{0}#rightarrowK#pi#pi#pi) }}";
        }
        else if (datafilepaths.Contains("LL")){
            // track = "LL";
            // cut_B_M_sb = "B_M > 5500 && B_M < 7000";
            latexInfo = "#splitline{#font[10]{LHCb Run 2, Track: LL}}{#font[10]{B^{-}#rightarrow#bar{p}#Lambda(D^{0}#rightarrowK#pi#pi#pi) }}";
        }
        
    }
    //------------------------------------------------------------
    // Run the RooFit using your original 'fit' function
    //------------------------------------------------------------
    Double_t* result = fit(
        dataChain,
        BDTcut,
        cut_D0,       // additional cuts
        fit_range,
        nBins,
        fig_name,
        "DNN Cut",
        latexInfo
    );

    double Nsig = result[0];
    double Nbkg = result[1];

    std::cout << "=========================================================\n";
    std::cout << "    Single BDT cut result\n";
    std::cout << "    BDT cut = " << BDTcut << "\n";
    std::cout << "    Signal   = " << Nsig << "\n";
    std::cout << "    Background = " << Nbkg << "\n";
    std::cout << "=========================================================\n";

    //------------------------------------------------------------
    // Save to text file
    //------------------------------------------------------------
    std::ofstream out(outfile);
    out << "BDTcut  Nsig  Nbkg\n";
    out << BDTcut << "  " << Nsig << "  " << Nbkg << "\n";
    out.close();

    //------------------------------------------------------------
    // Cleanup
    //------------------------------------------------------------
    delete[] result;
    delete dataChain;
    delete MCChain;

    return 0;
}



void Cut() {
    std::cout << "User Test 004: Classification Evaluation + ROC" << std::endl;
    FoMSingleCut();
    std::cout << "✔ DNN Cut applied Sucessfully" << std::endl;
}
