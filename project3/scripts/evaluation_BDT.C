#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TGraph.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TPaveText.h>
#include <iostream>
#include <TSystem.h>
using namespace std;

void evaluation() {
    cout << "User Test 021: Classification Evaluation + ROC" << endl;
    // -------------------------------------------------
    // USER-CHOSEN CHANNEL 
    // -------------------------------------------------
    TString channel = "D2H_DD";  
    // Example:
    // TString channel = "D4H_LL";
    // TString channel = "D2H_DD";
    // TString channel = "D4H_DD";

    cout << "Evaluating channel: " << channel << endl;

    // -------------------------------------------------
    // File + dataset paths now auto-generated
    // -------------------------------------------------
    TString fileName = Form(
        "/home/alexanum/WORKSPACE/testing/ANALYSIS/Output/B_BDT/TMVAClassification/TMVA_%s.root",
        channel.Data()
    );

    TFile *f = TFile::Open(fileName);
    if (!f || f->IsZombie()) {
        cout << "ERROR: Cannot open file " << fileName << endl;
        return;
    }

    // // Get optimal BDT cut
    // TH1F *hFOM = (TH1F*) f->Get(
    // Form("Method_BDT/BDT/MVA_BDT_SoverSqrtSplusB")
    // );

    // double optimalCut = 0.0;

    // if (hFOM) {
    //     int maxBin = hFOM->GetMaximumBin();
    //     optimalCut = hFOM->GetXaxis()->GetBinCenter(maxBin);
    //     cout << "TMVA optimal BDT cut (S/sqrt(S+B)) = "
    //         << optimalCut << endl;
    // } else {
    //     cout << "WARNING: TMVA FOM histogram not found!" << endl;
    // }


    // Auto dataset location
    TString datasetPath = Form("dataset_%s", channel.Data());
    TTree *train = (TTree*) f->Get(datasetPath + "/TrainTree");
    TTree *test  = (TTree*) f->Get(datasetPath + "/TestTree");

    if (!train || !test) {
        cout << "ERROR: Cannot find TrainTree/TestTree in " << datasetPath << endl;
        return;
    }

    Int_t nb = 40;
    Float_t xmin = -0.7;   //-0.7
    Float_t xmax =  0.7; //0.7

    TH1F *hSigTrain = new TH1F("hSigTrain", "", nb, xmin, xmax);
    TH1F *hSigTest  = new TH1F("hSigTest",  "", nb, xmin, xmax);
    TH1F *hBkgTrain = new TH1F("hBkgTrain", "", nb, xmin, xmax);
    TH1F *hBkgTest  = new TH1F("hBkgTest",  "", nb, xmin, xmax);

    train->Draw("BDT>>hSigTrain", "classID==0");
    test ->Draw("BDT>>hSigTest",  "classID==0");
    train->Draw("BDT>>hBkgTrain", "classID==1");
    test ->Draw("BDT>>hBkgTest",  "classID==1");

    hSigTrain->Scale(1.0 / hSigTrain->Integral());
    hSigTest ->Scale(1.0 / hSigTest->Integral());
    hBkgTrain->Scale(1.0 / hBkgTrain->Integral());
    hBkgTest ->Scale(1.0 / hBkgTest->Integral());

    hSigTest->SetFillColor(kBlue-7);
    hSigTest->SetFillStyle(1001);
    hSigTest->SetLineColor(kBlue+2);
    hSigTest->SetStats(0);

    hBkgTest->SetFillColor(kRed-7);
    hBkgTest->SetFillStyle(3354);
    hBkgTest->SetLineColor(kRed+1);
    hBkgTest->SetStats(0);

    hSigTrain->SetMarkerColor(kBlue+2);
    hSigTrain->SetMarkerStyle(20);
    hSigTrain->SetLineColor(kBlue+2);
    hSigTrain->SetStats(0);

    hBkgTrain->SetMarkerColor(kRed+1);
    hBkgTrain->SetMarkerStyle(21);
    hBkgTrain->SetLineColor(kRed+1);
    hBkgTrain->SetStats(0);

    double ksSig = hSigTrain->KolmogorovTest(hSigTest);
    double ksBkg = hBkgTrain->KolmogorovTest(hBkgTest);

    TH1F hSig("hSig", "", 200, xmin, xmax);
    TH1F hBkg("hBkg", "", 200, xmin, xmax);

    test->Draw("BDT>>hSig", "classID==0");
    test->Draw("BDT>>hBkg", "classID==1");

    double sigTot = hSig.Integral();
    double bkgTot = hBkg.Integral();

    TGraph *roc = new TGraph();
    roc->SetLineWidth(4);
    roc->SetLineColor(kBlue+3);
    int n = 200;
    double auc = 0;
    for (int i=1; i<=n; i++) {
        double effSig = hSig.Integral(i, n) / sigTot;
        double effBkg = hBkg.Integral(i, n) / bkgTot;   
        roc->SetPoint(i-1, effSig, 1.0 - effBkg);     //roc->SetPoint(i-1, effSig, 1 - effBkg);
    }
    // Compute AUC using trapezoidal rule
    for (int i = 1; i < n; i++) {
        double x0, y0, x1, y1;
        roc->GetPoint(i-1, x0, y0);
        roc->GetPoint(i, x1, y1);
        auc += 0.5 * (y0 + y1) * (x1 - x0);
    }
    auc = -auc;
    // double auc = roc->Integral() / roc->GetXaxis()->GetXmax();
    // double auc = roc->Integral();

    TCanvas *c = new TCanvas("c", "BDT Evaluation + ROC", 1100, 950);

    TPad *pad1 = new TPad("pad1","pad1",0,0.3,1,0.95);
    pad1->SetBottomMargin(0.15);
    pad1->SetTopMargin(0.22);
    pad1->SetLeftMargin(0.12);
    pad1->SetRightMargin(0.05);
    pad1->Draw();
    pad1->cd();

    TPaveText *titleBox = new TPaveText(0.0, 1, 1.0, 1.2, "NDC");
    titleBox->SetFillColor(kGray+1);
    titleBox->SetTextColor(kWhite);
    titleBox->SetTextFont(42);
    titleBox->SetBorderSize(0);
    titleBox->AddText("TMVA Overtraining Check for Classifier: BDT");
    titleBox->AddText(Form("Channel: %s", channel.Data()));
    titleBox->Draw();

    hSigTest->SetTitle("");
    hSigTest->GetXaxis()->SetTitle("BDT response");
    hSigTest->GetYaxis()->SetTitle("(1/N) dN / dx");
    hSigTest->GetYaxis()->SetTitleOffset(1.2);
    hSigTest->GetXaxis()->SetLabelSize(0.04); 
    hSigTest->GetYaxis()->SetLabelSize(0.04); 
    hSigTest->GetXaxis()->SetTitleSize(0.045); 
    hSigTest->GetYaxis()->SetTitleSize(0.045);
    float ymax = TMath::Max(hSigTest->GetMaximum(), hBkgTest->GetMaximum()) * 1.2;
    hSigTest->SetMaximum(ymax);


    // TLine *cutLine = new TLine(optimalCut, 0, optimalCut, ymax);
    // cutLine->SetLineColor(kBlack);
    // cutLine->SetLineWidth(3);
    // cutLine->SetLineStyle(2);
    // cutLine->Draw("SAME");

    // TLatex cutLabel;
    // cutLabel.SetTextSize(0.04);
    // cutLabel.DrawLatex(optimalCut + 0.02*ymax, 0.9*ymax, Form("TMVA cut = %.3f", optimalCut));

    hSigTest->Draw("HIST");
    hBkgTest->Draw("HIST SAME");
    hSigTrain->Draw("P SAME");
    hBkgTrain->Draw("P SAME");

    TLegend *leg = new TLegend(0.14, 0.79, 0.88, 0.88);
    leg->SetNColumns(2);
    leg->SetBorderSize(1);
    leg->SetFillColor(kWhite);
    leg->SetTextSize(0.035);
    leg->AddEntry(hSigTest,  "SIG (test sample)",     "f");
    leg->AddEntry(hSigTrain, "SIG (training sample)", "p");
    leg->AddEntry(hBkgTest,  "BKG (test sample)",     "f");
    leg->AddEntry(hBkgTrain, "BKG (training sample)", "p");
    leg->Draw();

    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.048);
    latex.DrawLatex(0.14, 0.94, Form("BDT Classification: %s", channel.Data()));
    latex.DrawLatex(0.14, 0.9,
        Form("Kolmogorov-Smirnov test: SIG (BKG) probability = %.3f (%.3f)",
             ksSig, ksBkg));

    c->cd();
    TPad *pad2 = new TPad("pad2","pad2",0,0.0,1,0.33);
    pad2->SetTopMargin(0.05);
    pad2->SetBottomMargin(0.30);
    pad2->SetLeftMargin(0.12);
    pad2->SetRightMargin(0.05);
    pad2->Draw();
    pad2->cd();

    roc->SetTitle("");
    roc->GetYaxis()->SetTitle("BKG rejection");
    roc->GetXaxis()->SetTitle("SIG efficiency");
    roc->GetXaxis()->SetTitleSize(0.09); 
    roc->GetYaxis()->SetTitleSize(0.09); 
    roc->GetXaxis()->SetLabelSize(0.06); 
    roc->GetYaxis()->SetLabelSize(0.06); 
    roc->GetXaxis()->SetTitleOffset(0.8); 
    roc->GetYaxis()->SetTitleOffset(0.55);

    roc->GetXaxis()->SetRangeUser(0.0, 1.0); 
    roc->GetYaxis()->SetRangeUser(0.0, 1.0);
    roc->Draw("AL");

    TLatex rocTxt;
    rocTxt.SetNDC();
    rocTxt.SetTextSize(0.07);
    rocTxt.DrawLatex(0.4,0.80,Form("ROC AUC = %.3f", auc));

    gSystem->Exec("mkdir -p /home/alexanum/WORKSPACE/testing/ANALYSIS/Output/Evaluation/");

    c->SaveAs(Form("/home/alexanum/WORKSPACE/testing/ANALYSIS/Output/Evaluation/BDT_overtraining_ROC_%s.png",
                   channel.Data()));

    cout << "✔ Plot created for channel " << channel << endl;
}
