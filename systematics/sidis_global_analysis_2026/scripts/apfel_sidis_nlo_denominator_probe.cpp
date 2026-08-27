// Exploratory APFEL++ SIDIS NLO numerator/denominator probe.
//
// This program is intentionally outside the Python production path.  It
// evaluates the massless NLO SIDIS F2/FL coefficient operators and the NLO
// inclusive DIS F2/FL denominator on the same LHAPDF PDF member.  The output
// is a row-level ratio table consumed by the isolated Python joint-fit driver.
// It is not a production prediction: bin integration, scale variations,
// heavy-quark treatment, and covariance construction still require closure.

#include <apfel/SIDIS.h>
#include <apfel/apfelxx.h>
#include <apfel/rotations.h>
#include <apfel/structurefunctionbuilder.h>
#include <LHAPDF/LHAPDF.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) out.push_back(field);
  return out;
}

struct Row {
  std::string id;
  std::string hadron;
  std::string charge;
  double x{};
  double z{};
  double q{};
  double y{};
};

double sidis_operator(const apfel::DoubleObject<apfel::Operator>& object,
                      const apfel::Distribution& first,
                      const apfel::Distribution& second,
                      double x, double z) {
  double result = 0.0;
  for (const auto& term : object.GetTerms())
    result += term.coefficient * (term.object1 * first).Evaluate(x)
              * (term.object2 * second).Evaluate(z);
  return result;
}
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: apfel_sidis_nlo_denominator_probe input.csv output.csv\n";
    return 2;
  }
  const std::string input = argv[1];
  const std::string output = argv[2];
  const std::string pdf_name = std::getenv("SIDIS_PDF")
      ? std::getenv("SIDIS_PDF") : "NNPDF40_nlo_as_01180";
  const std::string ff_prefix = std::getenv("SIDIS_FF_PREFIX")
      ? std::getenv("SIDIS_FF_PREFIX") : "NNFF10";

  auto pdf = LHAPDF::mkPDF(pdf_name, 0);
  std::map<std::string, LHAPDF::PDF*> ffs;
  ffs["pi+"] = LHAPDF::mkPDF((ff_prefix + "_PIp_nlo").c_str(), 0);
  ffs["pi-"] = LHAPDF::mkPDF((ff_prefix + "_PIm_nlo").c_str(), 0);
  ffs["K+"] = LHAPDF::mkPDF((ff_prefix + "_KAp_nlo").c_str(), 0);
  ffs["K-"] = LHAPDF::mkPDF((ff_prefix + "_KAm_nlo").c_str(), 0);

  const apfel::Grid gx({apfel::SubGrid(80, 1e-5, 3), apfel::SubGrid(40, 0.5, 3)});
  const apfel::Grid gz({apfel::SubGrid(80, 1e-3, 3), apfel::SubGrid(40, 0.8, 3)});
  const std::vector<double> thresholds = {0, 0, 0, 1.51, 4.92, 172.5};
  const auto sidis = apfel::InitializeSIDIS(gx, gz, thresholds);
  const auto f2_object = apfel::InitializeF2NCObjectsZM(gx, thresholds);
  const auto fl_object = apfel::InitializeFLNCObjectsZM(gx, thresholds);
  const std::vector<int> pids = {-5, -4, -3, -2, -1, 1, 2, 3, 4, 5};
  const std::map<int, double> charge2 = {
      {-5, 1. / 9}, {-4, 4. / 9}, {-3, 1. / 9}, {-2, 4. / 9}, {-1, 1. / 9},
      {1, 1. / 9}, {2, 4. / 9}, {3, 1. / 9}, {4, 4. / 9}, {5, 1. / 9}};
  const std::map<int, int> neutron_swap = {
      {1, 2}, {2, 1}, {-1, -2}, {-2, -1}, {3, 3}, {-3, -3},
      {4, 4}, {-4, -4}, {5, 5}, {-5, -5}, {6, 6}, {-6, -6}};

  // Build inclusive DIS observables through APFEL's Observable interface.
  // This path includes the pure coefficient-function and PDF-evolution NLO
  // terms; the lower-level direct overload intentionally used for the SIDIS
  // diagnostic does not provide that complete sum.
  const auto input_pdf = [&](double xx, double qq) {
    std::map<int, double> physical;
    physical[0] = pdf->xfxQ(21, xx, qq);
    for (int pid = 1; pid <= 6; ++pid) {
      physical[pid] = 0.5 * (pdf->xfxQ(pid, xx, qq)
                             + pdf->xfxQ(neutron_swap.at(pid), xx, qq));
      physical[-pid] = 0.5 * (pdf->xfxQ(-pid, xx, qq)
                              + pdf->xfxQ(neutron_swap.at(-pid), xx, qq));
    }
    return apfel::PhysToQCDEv(physical);
  };
  const auto alpha_s = [&](double qq) { return pdf->alphasQ(qq); };
  const auto ew_charges = [&](double qq) { return apfel::ElectroWeakCharges(qq, false); };
  const auto f2_observable = apfel::BuildStructureFunctions(
      f2_object, input_pdf, 1, alpha_s, ew_charges);
  const auto fl_observable = apfel::BuildStructureFunctions(
      fl_object, input_pdf, 1, alpha_s, ew_charges);

  std::ifstream stream(input);
  std::string line;
  if (!std::getline(stream, line)) throw std::runtime_error("empty input CSV");
  const auto header = split(line);
  std::map<std::string, int> index;
  for (int i = 0; i < static_cast<int>(header.size()); ++i) index[header[i]] = i;
  for (const auto& required : {"row_id", "hadron", "charge", "x", "z",
                               "Q_reconstructed", "y"})
    if (!index.count(required)) throw std::runtime_error(std::string("missing column ") + required);

  std::ofstream out(output);
  out << "row_id,hadron,charge,x,z,Q_reconstructed,y,lo_ratio,"
         "nlo_numerator_lo_den_ratio,nlo_full_den_ratio,f2_dis_nlo,fl_dis_nlo\n";
  out << std::setprecision(12);
  while (std::getline(stream, line)) {
    if (line.empty()) continue;
    const auto fields = split(line);
    if (fields.size() < header.size()) continue;
    Row row{fields[index["row_id"]], fields[index["hadron"]], fields[index["charge"]],
            std::stod(fields[index["x"]]), std::stod(fields[index["z"]]),
            std::stod(fields[index["Q_reconstructed"]]), std::stod(fields[index["y"]])};
    const std::string hadron_charge = row.hadron + row.charge;
    const auto ff = ffs.at(hadron_charge);
    std::map<int, apfel::Distribution> pdf_dist;
    std::map<int, apfel::Distribution> ff_dist;
    pdf_dist.emplace(0, apfel::Distribution(gx, [&](double xx) {
      return pdf->xfxQ(21, xx, row.q);
    }));
    // PhysToQCDEv requires all light/heavy +/- flavours, including top,
    // even though top is inactive and absent from the SIDIS charge sum.
    for (int pid = 1; pid <= 6; ++pid) {
      for (const int signed_pid : {pid, -pid}) {
        const int swap = neutron_swap.at(signed_pid);
        pdf_dist.emplace(signed_pid, apfel::Distribution(gx, [&, signed_pid, swap](double xx) {
          return 0.5 * (pdf->xfxQ(signed_pid, xx, row.q) + pdf->xfxQ(swap, xx, row.q));
        }));
      }
    }
    for (const int pid : pids) {
      const int swap = neutron_swap.at(pid);
      ff_dist.emplace(pid, apfel::Distribution(gz, [&, pid](double zz) {
        return ff->xfxQ(pid, zz, row.q);
      }));
    }
    ff_dist.emplace(21, apfel::Distribution(gz, [&](double zz) {
      return ff->xfxQ(21, zz, row.q);
    }));

    const double f2 = f2_observable.at(0).Evaluate(row.x, row.q);
    const double fl = fl_observable.at(0).Evaluate(row.x, row.q);

    double f20 = 0.0;
    double f21 = 0.0;
    double fl1 = 0.0;
    double denominator_lo = 0.0;
    for (const int pid : pids) {
      f20 += charge2.at(pid) * sidis_operator(sidis.C20qq, pdf_dist.at(pid), ff_dist.at(pid), row.x, row.z);
      f21 += charge2.at(pid) * sidis_operator(sidis.C21qq, pdf_dist.at(pid), ff_dist.at(pid), row.x, row.z);
      f21 += charge2.at(pid) * sidis_operator(sidis.C21qg, pdf_dist.at(pid), ff_dist.at(21), row.x, row.z);
      f21 += charge2.at(pid) * sidis_operator(sidis.C21gq, pdf_dist.at(pid), ff_dist.at(pid), row.x, row.z);
      fl1 += charge2.at(pid) * sidis_operator(sidis.CL1qq, pdf_dist.at(pid), ff_dist.at(pid), row.x, row.z);
      fl1 += charge2.at(pid) * sidis_operator(sidis.CL1qg, pdf_dist.at(pid), ff_dist.at(21), row.x, row.z);
      fl1 += charge2.at(pid) * sidis_operator(sidis.CL1gq, pdf_dist.at(pid), ff_dist.at(pid), row.x, row.z);
      denominator_lo += charge2.at(pid) * pdf_dist.at(pid).Evaluate(row.x);
    }
    const double as = pdf->alphasQ(row.q) / (4.0 * M_PI);
    const double yplus = 1.0 + (1.0 - row.y) * (1.0 - row.y);
    const double lo_ratio = f20 / (row.z * denominator_lo);
    const double nlo_num_lo_den = (yplus * (f20 + as * f21) - row.y * row.y * as * fl1)
                                  / (row.z * (yplus * denominator_lo));
    const double nlo_full_den = (yplus * (f20 + as * f21) - row.y * row.y * as * fl1)
                                / (row.z * (yplus * f2 - row.y * row.y * fl));
    out << row.id << ',' << row.hadron << ',' << row.charge << ',' << row.x << ',' << row.z << ','
        << row.q << ',' << row.y << ',' << lo_ratio << ',' << nlo_num_lo_den << ','
        << nlo_full_den << ',' << f2 << ',' << fl << '\n';
  }
  return 0;
}
