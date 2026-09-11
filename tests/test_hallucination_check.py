from hallucination_check import extract_numeric_claims, unsupported_numeric_claims


def test_extract_numeric_claims_finds_dollar_amount():
    assert "$50" in extract_numeric_claims("You'll get a $50 refund")


def test_extract_numeric_claims_finds_day_range():
    claims = extract_numeric_claims("This ships in 3-5 business days")
    assert any("3-5" in c or "business day" in c for c in claims)


def test_extract_numeric_claims_empty_for_no_numbers():
    assert extract_numeric_claims("Please DM us your order number") == set()


def test_unsupported_flags_number_not_in_context():
    reply = "You'll receive a $100 refund within 2 days"
    message = "my order never arrived"
    precedents = ["please DM your order number"]
    flagged = unsupported_numeric_claims(reply, message, precedents)
    assert "$100" in flagged


def test_unsupported_does_not_flag_number_present_in_precedent():
    reply = "You'll receive a refund within 3-5 business days"
    message = "where is my refund"
    precedents = ["refunds are processed within 3-5 business days"]
    flagged = unsupported_numeric_claims(reply, message, precedents)
    assert not any("3-5" in f for f in flagged)


def test_unsupported_does_not_flag_number_customer_already_stated():
    reply = "We see order #12345 and will look into it"
    message = "my order #12345 never arrived"
    precedents = []
    # order numbers aren't matched by the day/amount/percent regex at all,
    # so this should simply return no numeric claims
    flagged = unsupported_numeric_claims(reply, message, precedents)
    assert flagged == []
