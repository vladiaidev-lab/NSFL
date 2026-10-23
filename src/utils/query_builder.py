def query_parser(query):
    #templates = ['_ or _', '_ that are also both _ and _', '_ or _ or _', '_ that are not _', '_ that are also _', '_ that are also _ but not _', '_']
    original_query = query['original_query']
    full_query = query['query']
    clauses = {}
    atom_formulas = {}
    if query['metadata']['template'] == "_ that are also both _ and _":
        new_query_main_clause = original_query.split("</mark> that are also both <mark>")[0].replace("<mark>","")
        new_query_first_clause = original_query.split("</mark> that are also both <mark>")[1].split("</mark> and <mark>")[0]
        new_query_second_clause = original_query.split("</mark> that are also both <mark>")[1].split("</mark> and <mark>")[1].replace("</mark>","")
        #print(original_query)
        #new_query = "(" + new_query_main_clause + ") AND (" + new_query_first_and + ") AND (" + new_query_second_and +")"
        #clauses = [full_query, new_query_main_clause, 'that are also ' + new_query_first_clause,  'that are also ' + new_query_second_clause]
        clauses['atoms'] = {'atom0': new_query_main_clause, 'atom1': new_query_first_clause, 'atom2': new_query_second_clause}
        atom_formulas['atom_formulas'] = {"formula1": "atom0 & atom1 & atom2"}

    if query['metadata']['template'] == "_ or _":
        new_query_main_clause = original_query.split("</mark> or <mark>")[0].replace("<mark>","")
        new_query_second_clause = original_query.split("</mark> or <mark>")[1].replace("</mark>","")
        #print(original_query)
        #new_query = "(" + new_query_main_clause + ") OR (" + new_query_second_clause + ")" 
        #print(new_query)
        clauses['atoms'] = {'atom0': new_query_main_clause, 'atom1': new_query_second_clause}
        atom_formulas['atom_formulas'] = {"formula1": "atom0 | atom1"}

    if query['metadata']['template'] == "_ or _ or _":
        new_query_main_clause = original_query.split("</mark> or <mark>")[0].replace("<mark>","")
        new_query_second_clause = original_query.split("</mark> or <mark>")[1].replace("</mark>","")
        new_query_third_clause = original_query.split("</mark> or <mark>")[2].replace("</mark>","")
        #print(original_query)
        #new_query = "(" + new_query_main_clause + ") OR (" + new_query_second_clause +  ") OR (" + new_query_third_clause + ")"
        #print(new_query)
        clauses['atoms'] = {'atom0': new_query_main_clause, 'atom1': new_query_second_clause, 'atom2': new_query_third_clause}
        atom_formulas['atom_formulas'] = {"formula1": "atom0 | atom1 | atom2"}

    if query['metadata']['template'] == "_ that are not _":
        #print(original_query)
        new_query_main_clause = original_query.split("</mark> that are not <mark>")[0].replace("<mark>","")
        new_query_second_clause = original_query.split("</mark> that are not <mark>")[1].replace("</mark>","")
        #new_query = "(" + new_query_main_clause + ") NOT (" + new_query_second_clause + ")" 
        clauses['atoms'] = {'atom0': new_query_main_clause, 'atom1': new_query_second_clause}
        atom_formulas['atom_formulas'] = {"formula1": "atom0 & !atom1"}
        #print(new_query)
       
    if query['metadata']['template'] == "_ that are also _":
        #print(original_query)
        new_query_main_clause = original_query.split("</mark> that are also <mark>")[0].replace("<mark>","")
        new_query_second_clause = original_query.split("</mark> that are also <mark>")[1].replace("</mark>","")
        #new_query = "(" + new_query_main_clause + ") AND (" + new_query_second_clause + ")" 
        #print(new_query)
        clauses['atoms'] = {'atom0': new_query_main_clause, 'atom1': new_query_second_clause}
        atom_formulas['atom_formulas'] = {"formula1": "atom0 & atom1"}   
      
    if query['metadata']['template'] == "_ that are also _ but not _":
        new_query_main_clause = original_query.split("</mark> that are also <mark>")[0].replace("<mark>","")
        new_query_second_clause = original_query.split("</mark> that are also <mark>")[1].replace("</mark>","")
        new_query_third_clause = original_query.split("</mark> but not <mark>")[1].replace("</mark>","")
        #print(original_query)
        #new_query = "((" + new_query_main_clause + ") AND (" + new_query_second_clause +  ")) NOT (" + new_query_third_clause + ")"
        #print(new_query)
        clauses['atoms'] = {'atom0': new_query_main_clause, 'atom1': new_query_second_clause, 'atom2': new_query_third_clause}
        atom_formulas['atom_formulas'] = {"formula1": "atom0 & atom1 & !atom2"}

    if query['metadata']['template'] == "_":
        #print(original_query)
        #new_query = original_query.replace("</mark>", "").replace("<mark>","")
        clauses['atoms'] = {'atom0': full_query}
        atom_formulas['atom_formulas'] = {"formula1": "atom0"}

    return(clauses, atom_formulas)