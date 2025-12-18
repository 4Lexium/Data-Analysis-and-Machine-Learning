#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>

#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TString.h>
#include <TSystem.h>
#include <TROOT.h>
#include <TStopwatch.h>
#include <TMath.h>

#include <TMVA/Tools.h>
#include <TMVA/Reader.h>
#include <TMVA/MethodCuts.h>

#include <algorithm>
#include <TH2F.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TMatrixD.h>
#include <TStyle.h>
#include <TLegend.h>

using namespace TMVA;

int TMVAApplication_D2H_DD(TString datafilepath, TString datasetpath, TString TMVAApppath)
{
    // Load TMVA
    TMVA::Tools::Instance();
    std::cout << "\n==> Start TMVAClassificationApplication" << std::endl;

    TMVA::Reader *reader = new TMVA::Reader("!Color:!Silent");

    Float_t var_log_B_IPCHI2_OWNPV, var_log_B_ENDVERTEX_CHI2, var_log_D0_ENDVERTEX_CHI2;
    Float_t var_log_D0_FDCHI2_ORIVX, var_log_D0_K_IPCHI2_OWNPV, var_log_D0_pi1_IPCHI2_OWNPV;
    Float_t var_log_D0_pi1_PT, var_log_p_PT, var_MaxLogPT, var_MinLogPT;

    reader->AddVariable("log(B_IPCHI2_OWNPV)", &var_log_B_IPCHI2_OWNPV);
    reader->AddVariable("log(B_ENDVERTEX_CHI2)", &var_log_B_ENDVERTEX_CHI2);
    reader->AddVariable("log(D0_ENDVERTEX_CHI2)", &var_log_D0_ENDVERTEX_CHI2);
    reader->AddVariable("log(D0_FDCHI2_ORIVX)", &var_log_D0_FDCHI2_ORIVX);
    reader->AddVariable("log(D0_K_IPCHI2_OWNPV)", &var_log_D0_K_IPCHI2_OWNPV);
    reader->AddVariable("log(D0_pi1_IPCHI2_OWNPV)", &var_log_D0_pi1_IPCHI2_OWNPV);
    reader->AddVariable("log(D0_pi1_PT)", &var_log_D0_pi1_PT);
    reader->AddVariable("log(p_PT)", &var_log_p_PT);
    reader->AddVariable("max(log(Lambda_p_PT), log(Lambda_pi_PT))", &var_MaxLogPT);
    reader->AddVariable("min(log(Lambda_p_PT), log(Lambda_pi_PT))", &var_MinLogPT);

    TString methodName = "BDT method";
    TString weightfile = datasetpath + "/weights/TMVAClassification_BDT.weights.xml";
    reader->BookMVA(methodName, weightfile);

    // Load data
    TChain *datachain = new TChain("DecayTree");
    std::cout << "--- Using input file: " << datafilepath << std::endl;
    datachain->Add(datafilepath);

    Double_t uservar_B_IPCHI2_OWNPV, uservar_B_ENDVERTEX_CHI2, uservar_D0_ENDVERTEX_CHI2;
    Double_t uservar_D0_FDCHI2_ORIVX, uservar_D0_K_IPCHI2_OWNPV, uservar_D0_pi1_IPCHI2_OWNPV;
    Double_t uservar_D0_pi1_PT, uservar_p_PT, uservar_Lambda_p_PT, uservar_Lambda_pi_PT;

    datachain->SetBranchAddress("B_IPCHI2_OWNPV", &uservar_B_IPCHI2_OWNPV);
    datachain->SetBranchAddress("B_ENDVERTEX_CHI2", &uservar_B_ENDVERTEX_CHI2);
    datachain->SetBranchAddress("D0_ENDVERTEX_CHI2", &uservar_D0_ENDVERTEX_CHI2);
    datachain->SetBranchAddress("D0_FDCHI2_ORIVX", &uservar_D0_FDCHI2_ORIVX);
    datachain->SetBranchAddress("D0_K_IPCHI2_OWNPV", &uservar_D0_K_IPCHI2_OWNPV);
    datachain->SetBranchAddress("D0_pi1_IPCHI2_OWNPV", &uservar_D0_pi1_IPCHI2_OWNPV);
    datachain->SetBranchAddress("D0_pi1_PT", &uservar_D0_pi1_PT);
    datachain->SetBranchAddress("p_PT", &uservar_p_PT);
    datachain->SetBranchAddress("Lambda_p_PT", &uservar_Lambda_p_PT);
    datachain->SetBranchAddress("Lambda_pi_PT", &uservar_Lambda_pi_PT);

    Int_t nentries = datachain->GetEntries();
    std::cout << "--- Processing " << nentries << " events" << std::endl;
    TStopwatch sw; sw.Start();

    // Output file & tree
    TFile *outfile = new TFile(TMVAApppath, "RECREATE");
    TTree *outtree = datachain->CloneTree(0);
    Double_t Res_BDT; outtree->Branch("Res_BDT", &Res_BDT, "Res_BDT/D");

    // Histograms & correlation storage
    TH1F *h_BDT = new TH1F("h_BDT", "BDT output;BDT score;Events", 100, -1, 1);
    const int nVars = 10;
    std::vector<std::vector<double>> varsStorage(nVars);

    // Event loop
    for (Long64_t i = 0; i < nentries; i++) {
        if (i % 10000 == 0) std::cout << "--- Processing event: " << i << std::endl;
        datachain->GetEntry(i);

        var_log_B_IPCHI2_OWNPV = TMath::Log(uservar_B_IPCHI2_OWNPV);
        var_log_B_ENDVERTEX_CHI2 = TMath::Log(uservar_B_ENDVERTEX_CHI2);
        var_log_D0_ENDVERTEX_CHI2 = TMath::Log(uservar_D0_ENDVERTEX_CHI2);
        var_log_D0_FDCHI2_ORIVX = TMath::Log(uservar_D0_FDCHI2_ORIVX);
        var_log_D0_K_IPCHI2_OWNPV = TMath::Log(uservar_D0_K_IPCHI2_OWNPV);
        var_log_D0_pi1_IPCHI2_OWNPV = TMath::Log(uservar_D0_pi1_IPCHI2_OWNPV);
        var_log_D0_pi1_PT = TMath::Log(uservar_D0_pi1_PT);
        var_log_p_PT = TMath::Log(uservar_p_PT);
        var_MaxLogPT = TMath::Max(TMath::Log(uservar_Lambda_p_PT), TMath::Log(uservar_Lambda_pi_PT));
        var_MinLogPT = TMath::Min(TMath::Log(uservar_Lambda_p_PT), TMath::Log(uservar_Lambda_pi_PT));

        Res_BDT = reader->EvaluateMVA(methodName);
        outtree->Fill();
        h_BDT->Fill(Res_BDT);

        // Store for correlation
        double vars[nVars] = { var_log_B_IPCHI2_OWNPV, var_log_B_ENDVERTEX_CHI2, var_log_D0_ENDVERTEX_CHI2,
                               var_log_D0_FDCHI2_ORIVX, var_log_D0_K_IPCHI2_OWNPV, var_log_D0_pi1_IPCHI2_OWNPV,
                               var_log_D0_pi1_PT, var_log_p_PT, var_MaxLogPT, var_MinLogPT };
        for (int v = 0; v < nVars; v++) varsStorage[v].push_back(vars[v]);
    }

    // Compute correlation matrix using TMath::Mean
    TMatrixD corr(nVars, nVars);
    for (int i = 0; i < nVars; i++) {
        for (int j = 0; j < nVars; j++) {
            double mean_i = TMath::Mean(varsStorage[i].size(), varsStorage[i].data());
            double mean_j = TMath::Mean(varsStorage[j].size(), varsStorage[j].data());
            double num = 0, denom_i = 0, denom_j = 0;
            for (size_t k = 0; k < varsStorage[i].size(); k++) {
                num += (varsStorage[i][k] - mean_i) * (varsStorage[j][k] - mean_j);
                denom_i += TMath::Power(varsStorage[i][k] - mean_i, 2);
                denom_j += TMath::Power(varsStorage[j][k] - mean_j, 2);
            }
            corr(i, j) = num / TMath::Sqrt(denom_i * denom_j);
        }
    }

    gSystem->mkdir("/home/alexanum/WORKSPACE/testing/ANALYSIS/Output/B_BDT/TMVAApplication/Figures", kTRUE);

    // Detach histogram from file so it's not deleted when TFile is closed
    h_BDT->SetDirectory(0);

    // BDT output plot
    TCanvas *c1 = new TCanvas("c1", "BDT output", 800, 600);
    h_BDT->Draw();
    c1->SaveAs("/home/alexanum/WORKSPACE/testing/ANALYSIS/Output/B_BDT/TMVAApplication/Figures/D2H_DD_BDT_output.png");

    // Detach correlation histogram from file
    TH2F *hCorr = new TH2F("hCorr", "Variable correlation matrix", nVars, 0, nVars, nVars, 0, nVars);
    hCorr->SetDirectory(0);  // Detach from TFile
    for (int i = 0; i < nVars; i++)
        for (int j = 0; j < nVars; j++)
            hCorr->SetBinContent(i + 1, j + 1, corr(i, j));

    // Correlation matrix plot
    TCanvas *c2 = new TCanvas("c2", "Correlation matrix", 800, 600);
    hCorr->Draw("COLZ TEXT");
    c2->SaveAs("/home/alexanum/WORKSPACE/testing/ANALYSIS/Output/B_BDT/TMVAApplication/Figures/D2H_DD_CorrelationMatrix.png");
    std::cout<<"Plotting Done!" << std::endl;
    // Write output tree
    outtree->Write();
    outfile->Close();

    // Clean up
    delete h_BDT;
    delete hCorr;
    delete c1;
    delete c2;
    delete reader;

    sw.Stop();
    std::cout << "--- End of event loop: "; sw.Print();
    std::cout << "==> TMVAClassificationApplication is done!\n";

    return 0;
}
