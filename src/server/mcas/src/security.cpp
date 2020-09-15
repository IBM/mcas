#include "security.h"

#include <common/exceptions.h>
#include <common/logging.h>
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include <fstream>

namespace mcas
{
class Shard_security_state : common::log_source {
 public:
  Shard_security_state(const std::string &certs_path) : common::log_source(3), _cert(nullptr), _cert_base64{}
  {
    try {
      FILE *fp = fopen(certs_path.c_str(), "r");
      if (!fp) return;  // throw Constructor_exception("unable to open cert file");

      _cert = PEM_read_X509(fp, nullptr, nullptr /* passwd callback */, nullptr);
      if (_cert) {
        /* check certificate */
        STACK_OF(X509) *sk = sk_X509_new_null();
        sk_X509_push(sk, _cert);

        X509_NAME *subj = X509_get_subject_name(_cert);

        char cert_email[256];
        X509_NAME_get_text_by_NID(subj, NID_pkcs9_emailAddress, cert_email, 256);
        PLOG("Cert email: %s", cert_email);
        assert(std::string(cert_email) == "daniel.waddington@ibm.com");
        if (debug_level() > 3) {
          for (int i = 0; i < X509_NAME_entry_count(subj); i++) {
            X509_NAME_ENTRY *e = X509_NAME_get_entry(subj, i);
            ASN1_STRING *    d = X509_NAME_ENTRY_get_data(e);
            unsigned char *  c;
            if (ASN1_STRING_to_UTF8(&c, d)) {
              PLOG("X509: %s", c);
              OPENSSL_free(c);
            }
          }
        }

        CPLOG(0, "X509 certificate OK.");
      }

      fclose(fp);

      std::ifstream t(certs_path);
      _cert_base64.assign(std::istreambuf_iterator<char>(t), std::istreambuf_iterator<char>());
    }
    catch (...) {
      PLOG("unable to load certificate. disabling authentication");
      _cert = nullptr;
    }
  }

  Shard_security_state(const Shard_security_state &) = delete;
  Shard_security_state &operator=(const Shard_security_state &) = delete;

  ~Shard_security_state()
  {
    if (_cert) X509_free(_cert);
  }

 private:
  X509 *      _cert;
  std::string _cert_base64;
};

Shard_security::Shard_security(const std::string &certs_path)
    : _certs_path(certs_path),
      _auth_enabled(!_certs_path.empty()),
      _state(std::make_shared<Shard_security_state>(certs_path))
{
  if (debug_level() > 1) PMAJOR("Shard_security: auth=%s cert path (%s)", _auth_enabled ? "y" : "n", certs_path.c_str());

  // if(_auth_enabled) {
  //   PLOG("Auth enabled:");
  //   try {
  //     std::ifstream ifs(certs_path);
  //   }
  //   catch(...) {
  //     PLOG("unable to load certificate. disabling authentication");
  //     _auth_enabled = false;
  //   }
  // }
}

}  // namespace mcas
