/*
   Copyright [2017-2020] [IBM Corporation]
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"

#include <string>
#include <chrono>
#include <api/components.h>
#include <api/crypto_itf.h>
#include <common/cpu.h>
#include <common/str_utils.h>
#include <common/dump_utils.h>
#include <gtest/gtest.h>
#include <boost/program_options.hpp>

struct {
  std::string server_ip;
  int         server_port;
  std::string cert_file;
  std::string key_file;
  unsigned    debug_level;
} Options;

// component::ICrypto_factory *fact;
// Objects declared here can be used by all tests in the test case
static component::ICrypto *_crypto;

using namespace component;

static void global_init()
{
  /* create object instance through factory */
  component::IBase *comp = component::load_component("libcomponent-tls.so", tls_factory);

  ASSERT_TRUE(comp);
  auto fact = make_itf_ref(static_cast<ICrypto_factory *>(comp->query_interface(ICrypto_factory::iid())));

  std::map<std::string, std::string> params;
  _crypto = fact->create(Options.debug_level, params);
  ASSERT_TRUE(_crypto);
}

namespace
{
// The fixture for testing class Foo.
class TLS_test : public ::testing::Test {
 protected:
  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp()
  {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown()
  {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }
};

#define CIPHER_SUITE \
  "NORMAL:+AEAD"  // PERFORMANCE:+AEAD"
                  // //"PERFORMANCE:-VERS-SSL3.0:-VERS-TLS1.0:-VERS-TLS1.1:-ARCFOUR-128:-PSK:-DHE-PSK:+AEAD"

TEST_F(TLS_test, CertServer)
{
  ASSERT_TRUE(_crypto->initialize(CIPHER_SUITE, Options.cert_file, Options.key_file) == S_OK);

  auto session = _crypto->accept_cert_session(8888);

  std::string label = "SOMELABEL";
  PMAJOR("Server keys:");
  for (unsigned i = 0; i < 10; i++) {
    std::string key3;
    ASSERT_TRUE(_crypto->export_key(session, label, std::to_string(i), 8, key3) == S_OK);
    hexdump(key3.c_str(), 8);
  }

  size_t  data_len = 512;
  char    data[512];
  ssize_t rlen = _crypto->record_recv(session, data, data_len);

  PMAJOR("Got encrypted record!!!! (len=%lu)", rlen);
  hexdump(data, std::size_t(rlen));

  /* this is actually still cipher text - we just used record_send/recv for testing */

  std::string aead_key;
  ASSERT_TRUE(_crypto->export_key(session, label, "some_context", 16, aead_key) == S_OK);
  ASSERT_TRUE(_crypto->initialize_cipher(session, ICrypto::CIPHER_AES_128_GCM, aead_key) == S_OK);

  char   out_ptext[512];
  size_t out_ptext_len = 512;
  int    nonce         = 1;

  ASSERT_TRUE(_crypto->aead_decrypt(session, &nonce, sizeof(nonce), nullptr, /* optional extra auth data */
                                    0, data, std::size_t(rlen), out_ptext, &out_ptext_len) == S_OK);
  PMAJOR("Decrypted: (%lu) %s", out_ptext_len, out_ptext);
  ASSERT_TRUE(out_ptext_len == 23);
  ASSERT_TRUE(strcmp(out_ptext, "This is my secret text!") == 0);

  /* close session */
  ASSERT_TRUE(_crypto->close_session(session) == S_OK);
}

TEST_F(TLS_test, CertClient)
{
  auto session =
      _crypto->open_cert_session(CIPHER_SUITE, "127.0.0.1", 8888, "dwaddington", Options.cert_file, Options.key_file);

  char        key[8] = {0};
  std::string label  = "SOMELABEL";
  ASSERT_TRUE(_crypto->export_key(session, label, "" /* context is optional */, 8, key) == S_OK);
  PMAJOR("Got key material:");
  hexdump(key, 8);

  std::string key2;
  ASSERT_TRUE(_crypto->export_key(session, label, "some_context", 16, key2) == S_OK);
  PMAJOR("Got key material:");
  hexdump(key2.c_str(), 16);

  /* check these are the same both sides */
  PMAJOR("Client keys:");
  for (unsigned i = 0; i < 10; i++) {
    std::string key3;
    ASSERT_TRUE(_crypto->export_key(session, label, std::to_string(i), 8, key3) == S_OK);
    hexdump(key3.c_str(), 8);
  }

  std::string payload = "This is some payload";
  std::string digest;
  ASSERT_TRUE(_crypto->hmac(component::ICrypto::mac_algorithm_t::MAC_SHA256, key2, payload.data(), payload.size(),
                            digest) == S_OK);
  PMAJOR("HMAC digest (len=%lu):", digest.size());
  hexdump(digest.data(), digest.size());
  ASSERT_TRUE(digest.size() == 256 / 8);

  {
    const unsigned           rounds = 10000;
    std::vector<std::string> data;
    std::vector<std::string> keys;

    for (unsigned i = 0; i < rounds; i++) {
      data.push_back(common::random_string(4096));
      keys.push_back(common::random_string(32));
    }

    __sync_synchronize();

    using clock     = std::chrono::high_resolution_clock;
    auto start_time = clock::now();

    std::string out_digest;
    for (unsigned i = 0; i < rounds; i++) {
      ASSERT_TRUE(_crypto->hmac(component::ICrypto::mac_algorithm_t::MAC_SHA256, keys[i], data[i].data(),
                                data[i].size(), out_digest) == S_OK);
    }

    __sync_synchronize();

    auto   secs    = std::chrono::duration<double>(clock::now() - start_time).count();
    double per_sec = double(rounds) / secs;

    PINF("Time: %.2f sec", secs);
    PINF("Rate: %.0f pages/sec", per_sec);
  }

  /* note here, key is 128 bit to match cipher */
  std::string aead_key;
  ASSERT_TRUE(_crypto->export_key(session, label, "some_context", 16, aead_key) == S_OK);
  ASSERT_TRUE(_crypto->initialize_cipher(session, ICrypto::CIPHER_AES_128_GCM, aead_key) == S_OK);

  /* test cipher encrypt function */
  std::string ptext = "This is my secret text!";
  uint64_t    nonce = 1;
  char        ctext[512];
  size_t      ctext_len = 512;
  ASSERT_TRUE(_crypto->aead_encrypt(session, &nonce, sizeof(nonce), nullptr, /* optional extra auth data */
                                    0, ptext.data(), ptext.size(), ctext, &ctext_len) == S_OK);

  PLOG("ptext_len=%lu ctext_len=%lu", ptext.size(), ctext_len);
  PLOG("nonce=%lu", nonce);
  PINF("Cipher text:");
  hexdump(ctext, ctext_len);

  /* of course we'd normally pass in plain text, but for this test we want to pass
     the cipher text. using record_send just makes it easier */
  ASSERT_TRUE(_crypto->record_send(session, ctext, ctext_len) == boost::numeric_cast<ssize_t>(ctext_len));

  /* clean up */
  ASSERT_TRUE(_crypto->close_session(session) == S_OK);
}

}  // namespace

int main(int argc, char **argv)
{
  namespace po = boost::program_options;

  ::testing::InitGoogleTest(&argc, argv);

  try {
    po::options_description desc("Options");
    desc.add_options()("help", "Help")("debug", po::value<unsigned>()->default_value(0), "Debug level 0-3")(
        "server", po::value<std::string>()->default_value("127.0.0.1"), "Server address IP")(
        "key", po::value<std::string>()->default_value("./dist/certs/mcas-privkey.pem"))(
        "cert", po::value<std::string>()->default_value("./dist/certs/mcas-cert.pem"))(
        "port", po::value<int>()->default_value(8888), "Server port");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    Options.server_ip   = vm["server"].as<std::string>();
    Options.server_port = vm["port"].as<int>();
    Options.cert_file   = vm["cert"].as<std::string>();
    Options.key_file    = vm["key"].as<std::string>();
    Options.debug_level = vm["debug"].as<unsigned>();
  }
  catch (...) {
  }

  global_init();

  return RUN_ALL_TESTS();
}

#pragma GCC diagnostic pop
