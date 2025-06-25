# Policy to allow reading secrets under the 'secret/' KV engine
path "secret/data/*" {
  capabilities = ["read", "list"]
}
