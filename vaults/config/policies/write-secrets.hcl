# Policy to allow creating and updating secrets under the 'secret/' KV engine
path "secret/data/*" {
  capabilities = ["create", "update"]
}
