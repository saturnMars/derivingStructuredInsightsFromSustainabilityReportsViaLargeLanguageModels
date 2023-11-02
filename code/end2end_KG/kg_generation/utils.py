from neo4j import GraphDatabase, basic_auth, TrustAll
from neo4j.exceptions import ServiceUnavailable
import logging
from string import punctuation

class AuraDBApp:

    def __init__(self, uri, user, password):
        self.__driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        print("SERVER INFO:")
        print('\n'.join([f"{key}: {value}"for key, value in self.__driver.get_server_info().__dict__.items()]), "\n")

    def close(self):
        self.__driver.close()
        
    def _query(self, query, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response
    
    def create_company_relationship(self, company, esg_topic, verbose = False):
        query = (
            "MERGE (root:Company { name: $node1 }) "
            "MERGE (leaft:ESG_topic { name: $node2 }) "
            "MERGE (root)-[r:REPORTED]->(leaft) "
            "RETURN root, leaft"
        )
        results = self._query(query, parameters={"node1": company, "node2": esg_topic})
        
        if results and verbose:
            for row in results:
                print(row["root"]["name"], "--> HAS -->", row["leaft"]["name"])

        return results
    
    def create_esg_relationship(self, esg_topic, predicate, esg_object, properties, verbose = True):
        clear_name_main = lambda text: text.translate(str.maketrans('', '', punctuation)).upper().replace(' ', '_').replace('’', "'")
        clear_name = lambda text: '_' + clear_name_main(text) if text[0].isdigit() else clear_name_main(text)
        
        #print('\npredicate:', predicate)
        str_properties = ', '.join([key.lstrip("_") + ' :"' + value.replace("\"", "'").replace('’', "'") + '"' for key, value in properties.items()])
        
        query = (
            "MERGE (root:ESG_topic { name: $node1 }) "
            "MERGE (leaft:Object { _name: $node2, " + str_properties + "}) "
            f"MERGE (root)-[r: {clear_name(predicate)} ]->(leaft) "
            "RETURN root, leaft"
        )
        results = self._query(query, parameters={"node1": esg_topic, "node2": esg_object})
        
        if results and verbose:
            for row in results:
                print("\t", row["root"]["name"], "-->", clear_name(predicate), "-->", row["leaft"]["_name"])

        return results

    def find_person(self, person_name):
        with self.__driver.session(database="neo4j") as session:
            result = session.execute_read(self._find_and_return_person, person_name)
            for row in result:
                print("Found person: {row}".format(row=row))

    @staticmethod
    def _find_and_return_person(tx, person_name):
        query = (
            "MATCH (p:Person) "
            "WHERE p.name = $person_name "
            "RETURN p.name AS name"
        )
        result = tx.run(query, person_name=person_name)
        return [row["name"] for row in result]
    
    def clear_db(self):
        with self.__driver.session(database="neo4j") as session:
            result = session.execute_write(self._clear_db)
            
    @staticmethod
    def _clear_db(tx):
        query = (
            "MATCH (n) "
            "DETACH DELETE n"
        )
        result = tx.run(query)
        return result