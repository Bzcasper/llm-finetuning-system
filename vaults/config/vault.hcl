listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 1
}

storage "file" {
  path = "/vault/data"
}

# Default dev root token
disable_mlock = true
